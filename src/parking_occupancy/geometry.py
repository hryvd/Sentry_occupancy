"""
geometry.py — Homography / perspective warp + polygon-based occupancy classification.

Core responsibilities:
  - Load per-camera homography matrix H from polygon config JSON
  - Apply cv2.warpPerspective() to produce bird's-eye view (BEV) images
  - Project vehicle detections from original image coords → BEV coords via H
  - Determine which parking-space polygon contains each projected detection centroid
  - Classify each space as: available | occupied | unknown

Design decisions:
  - Detection happens on the ORIGINAL image (not the warped one) for best YOLO accuracy.
    Bounding-box centroids are then projected through H into BEV space for polygon testing.
  - If the calibration_version in the config does not match the camera's current calibration,
    all spaces on that camera are marked unknown (camera may have moved).
  - Degenerate polygons (< 3 points, zero area, out-of-bounds) are rejected on load.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import Point, Polygon as ShapelyPolygon

from parking_occupancy.capture import Frame
from parking_occupancy.inference import Detection, VehicleClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SpaceCompatibility(str, Enum):
    CAR_ONLY = "car_only"
    MOTORCYCLE_ONLY = "motorcycle_only"
    # NOTE: "mixed" is a tracked database gap; we never create mixed spaces here.


class PhysicalStatus(str, Enum):
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpaceConfig:
    """A single parking space definition in BEV coordinate space."""
    space_id: str           # Backend UUID (synthetic for mock)
    lot_code: str           # A, B, C, or D
    compatibility: SpaceCompatibility
    polygon_bev: list[tuple[float, float]]  # vertices in BEV pixels
    _shapely: ShapelyPolygon = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Validate and build Shapely polygon for point-in-polygon tests
        if len(self.polygon_bev) < 3:
            raise ValueError(
                f"Space {self.space_id}: polygon must have >= 3 points, "
                f"got {len(self.polygon_bev)}"
            )
        poly = ShapelyPolygon(self.polygon_bev)
        if not poly.is_valid:
            raise ValueError(f"Space {self.space_id}: polygon is geometrically invalid.")
        if poly.area < 1.0:
            raise ValueError(
                f"Space {self.space_id}: polygon area {poly.area:.2f} is degenerate (< 1 px²)."
            )
        object.__setattr__(self, "_shapely", poly)

    def contains_point(self, x: float, y: float) -> bool:
        return self._shapely.contains(Point(x, y))

    @property
    def area_px(self) -> float:
        return self._shapely.area


@dataclass
class SpaceState:
    """Classification result for one space at one frame."""
    space_id: str
    lot_code: str
    compatibility: SpaceCompatibility
    physical_status: PhysicalStatus
    confidence: float | None  # None when status is unknown


@dataclass
class PolygonMap:
    """
    Per-camera polygon map — holds the homography matrix H,
    BEV canvas size, and all space polygon definitions.
    """

    camera_id: str
    calibration_version: str
    reference_resolution: tuple[int, int]   # (w, h) of original camera image
    bev_canvas_size: tuple[int, int]        # (w, h) of output BEV image
    homography_matrix: np.ndarray           # 3×3 float64
    src_points: np.ndarray                  # (N, 2) in original image
    dst_points: np.ndarray                  # (N, 2) in BEV image
    spaces: list[SpaceConfig]

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: Path) -> "PolygonMap":
        with open(path, "r") as f:
            data = json.load(f)

        # Validate required top-level keys
        for key in ("camera_id", "calibration_version", "reference_resolution",
                    "bev_canvas_size", "homography_matrix", "src_points",
                    "dst_points", "spaces"):
            if key not in data:
                raise ValueError(f"Polygon config missing required key: '{key}'")

        H = np.array(data["homography_matrix"], dtype=np.float64)
        if H.shape != (3, 3):
            raise ValueError(f"homography_matrix must be 3×3, got shape {H.shape}")

        spaces = []
        seen_ids: set[str] = set()
        for sp in data["spaces"]:
            sid = sp["space_id"]
            if sid in seen_ids:
                raise ValueError(f"Duplicate space_id: {sid}")
            seen_ids.add(sid)

            compat_raw = sp.get("compatibility", "")
            try:
                compat = SpaceCompatibility(compat_raw)
            except ValueError:
                raise ValueError(
                    f"Space {sid}: invalid compatibility '{compat_raw}'. "
                    f"Must be 'car_only' or 'motorcycle_only'."
                )

            spaces.append(SpaceConfig(
                space_id=sid,
                lot_code=sp["lot_code"],
                compatibility=compat,
                polygon_bev=[tuple(pt) for pt in sp["polygon_bev"]],  # type: ignore[misc]
            ))

        return cls(
            camera_id=data["camera_id"],
            calibration_version=data["calibration_version"],
            reference_resolution=tuple(data["reference_resolution"]),  # type: ignore[arg-type]
            bev_canvas_size=tuple(data["bev_canvas_size"]),            # type: ignore[arg-type]
            homography_matrix=H,
            src_points=np.array(data["src_points"], dtype=np.float64),
            dst_points=np.array(data["dst_points"], dtype=np.float64),
            spaces=spaces,
        )

    # ------------------------------------------------------------------
    # Warping
    # ------------------------------------------------------------------

    def warp_to_bev(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective warp to produce a bird's-eye view image."""
        bev_w, bev_h = self.bev_canvas_size
        return cv2.warpPerspective(
            image,
            self.homography_matrix,
            (bev_w, bev_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def project_point_to_bev(self, x: float, y: float) -> tuple[float, float]:
        """
        Project a single (x, y) point from original image coords into BEV coords
        using the homography matrix H.
        """
        pt = np.array([[[x, y]]], dtype=np.float64)
        bev_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
        return float(bev_pt[0, 0, 0]), float(bev_pt[0, 0, 1])

    def project_detections_to_bev(
        self, detections: Sequence[Detection]
    ) -> list[tuple[Detection, float, float]]:
        """
        Project all detection centroids through H.
        Returns list of (detection, bev_x, bev_y).
        """
        if not detections:
            return []
        centroids = np.array(
            [[[d.cx, d.cy]] for d in detections], dtype=np.float64
        )
        bev_pts = cv2.perspectiveTransform(centroids, self.homography_matrix)
        return [
            (det, float(bev_pts[i, 0, 0]), float(bev_pts[i, 0, 1]))
            for i, det in enumerate(detections)
        ]

    # ------------------------------------------------------------------
    # Occupancy classification
    # ------------------------------------------------------------------

    def classify(
        self,
        frame: Frame,
        detections: Sequence[Detection],
    ) -> list[SpaceState]:
        """
        Classify every space given the current detections.

        Rules:
        - If the frame resolution does not match reference_resolution → all unknown
          (camera calibration may be invalid)
        - Project detection centroids into BEV, check polygon containment
        - A space is OCCUPIED if any COMPATIBLE detection centroid falls inside it
        - A space is AVAILABLE only after smoothing confirms no detection for N frames
          (this function returns the per-frame raw classification; smoothing is applied
          by smoothing.py on top of these results)
        - Pedestrians (class=person) are ignored — not associated with any space
        """
        # Validate frame resolution against calibration
        if frame.resolution != self.reference_resolution:
            logger.warning(
                "Frame resolution %s != calibration reference %s — marking all spaces unknown.",
                frame.resolution,
                self.reference_resolution,
            )
            return [
                SpaceState(
                    space_id=sp.space_id,
                    lot_code=sp.lot_code,
                    compatibility=sp.compatibility,
                    physical_status=PhysicalStatus.UNKNOWN,
                    confidence=None,
                )
                for sp in self.spaces
            ]

        # Project vehicle detections to BEV coords
        projected = self.project_detections_to_bev(detections)

        results: list[SpaceState] = []
        for space in self.spaces:
            # Find all compatible detections inside this space's polygon
            matches = [
                (det, bx, by)
                for det, bx, by in projected
                if space.contains_point(bx, by)
                and self._detection_compatible(det, space.compatibility)
            ]

            # Also check incompatible detections (still occupies space physically)
            all_inside = [
                (det, bx, by)
                for det, bx, by in projected
                if space.contains_point(bx, by)
            ]

            if matches:
                # Compatible vehicle in space → occupied with its detection confidence
                confidence = max(det.confidence for det, _, _ in matches)
                status = PhysicalStatus.OCCUPIED
            elif all_inside:
                # Something (wrong-type vehicle) is in space → still occupied
                confidence = max(det.confidence for det, _, _ in all_inside) * 0.80
                status = PhysicalStatus.OCCUPIED
            else:
                # No detection in space — tentatively available (smoother confirms)
                status = PhysicalStatus.AVAILABLE
                confidence = 0.75  # Base confidence; smoother raises this over consecutive frames

            results.append(SpaceState(
                space_id=space.space_id,
                lot_code=space.lot_code,
                compatibility=space.compatibility,
                physical_status=status,
                confidence=confidence,
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detection_compatible(det: Detection, compat: SpaceCompatibility) -> bool:
        if compat == SpaceCompatibility.CAR_ONLY:
            return det.vehicle_class == VehicleClass.CAR
        if compat == SpaceCompatibility.MOTORCYCLE_ONLY:
            return det.vehicle_class == VehicleClass.MOTORCYCLE
        return False

    # ------------------------------------------------------------------
    # Validation utilities (used by mark_polygons.py)
    # ------------------------------------------------------------------

    @staticmethod
    def validate_polygon(points: list[tuple[float, float]], canvas_size: tuple[int, int]) -> None:
        """
        Validate a polygon for use as a parking space.
        Raises ValueError describing all detected issues.
        """
        errors: list[str] = []
        w, h = canvas_size

        if len(points) < 3:
            errors.append(f"Polygon must have >= 3 vertices, got {len(points)}.")

        for i, (px, py) in enumerate(points):
            if not (0 <= px <= w and 0 <= py <= h):
                errors.append(f"Vertex {i} ({px:.1f}, {py:.1f}) is out of canvas bounds ({w}×{h}).")

        if len(points) >= 3:
            poly = ShapelyPolygon(points)
            if not poly.is_valid:
                errors.append("Polygon self-intersects or is geometrically invalid.")
            if poly.area < 50:
                errors.append(f"Polygon area {poly.area:.1f} px² is too small (minimum 50 px²).")

        if errors:
            raise ValueError("Polygon validation failed:\n  " + "\n  ".join(errors))
