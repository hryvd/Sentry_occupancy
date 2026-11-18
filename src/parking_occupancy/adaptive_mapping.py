"""
adaptive_mapping.py — Module 1B: Adaptive Slot Mapping.

Sits between Geometry/Warping (Module 1) and Contextual Awareness (Module 2).
Responsibilities:
  1. Anchor-guided micro-drift compensation (delta_H) with hard safety threshold (tau_drift).
  2. Row-corridor auto-partitioning (painted-line snapping + uniform metric fallback).
  3. Effective boundary adaptation (strict_stalls vs. adaptive_curb).
  4. Perspective-depth scale & distortion factor (J_i) via local Jacobian of H⁻¹.

Guardrails:
  - Never creates new space_ids beyond pre-registered spaces.
  - Never creates 'mixed' compatibility spaces.
  - If drift > tau_drift or anchors are lost, marks affected spaces UNKNOWN.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon as ShapelyPolygon, box

from parking_occupancy.capture import Frame
from parking_occupancy.geometry import (
    CameraAngleType,
    PolygonMap,
    ProjectedVehicleFootprint,
    SpaceCompatibility,
    SpaceConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & Data Models
# ---------------------------------------------------------------------------

class RowPolicyMode(str, Enum):
    STRICT_STALLS = "strict_stalls"
    ADAPTIVE_CURB = "adaptive_curb"


@dataclass(frozen=True)
class GroundAnchor:
    """A static vehicle-free ground landmark used to track camera micro-drift."""
    anchor_id: str
    image_xy: tuple[float, float]
    bev_xy: tuple[float, float]


@dataclass(frozen=True)
class RowCorridorConfig:
    """
    Defines a calibrated row quadrilateral containing K pre-registered slots
    ordered along the row axis from start edge to end edge.
    """
    row_id: str
    space_ids: list[str]
    compatibility: SpaceCompatibility
    quad_bev: list[tuple[float, float]]  # [top-left, top-right, bottom-right, bottom-left]
    policy_mode: RowPolicyMode = RowPolicyMode.STRICT_STALLS
    min_vehicle_width_px: float = 42.0


@dataclass
class AdaptiveSlotGeometry:
    """Per-slot adapted geometry passed to Contextual Awareness & Fusion Model."""
    space_id: str
    slot_number: str
    lot_code: str
    compatibility: SpaceCompatibility
    policy_mode: RowPolicyMode
    baseline_polygon: ShapelyPolygon
    effective_polygon: ShapelyPolygon
    slot_area: float
    effective_area: float
    usable_width_px: float
    depth_distortion_j: float
    is_valid_coverage: bool = True
    drift_adapted: bool = False
    unlined_slot_blocked: bool = False
    status_reason: str = "nominal"


@dataclass
class AdaptiveMapResult:
    """Frame-level output from AdaptiveSlotMapper."""
    calibration_version: str
    camera_angle_type: CameraAngleType
    effective_homography: np.ndarray
    drift_px: float
    drift_adapted: bool
    camera_moved_fault: bool
    slots: list[AdaptiveSlotGeometry]


# ---------------------------------------------------------------------------
# Main Adaptive Slot Mapper
# ---------------------------------------------------------------------------

class AdaptiveSlotMapper:
    """
    Dynamically refines slot boundaries, tracks micro-drift, and computes
    perspective depth distortion while preserving registered space_id invariants.
    """

    def __init__(
        self,
        polygon_map: PolygonMap,
        *,
        tau_drift_px: float = 10.0,
        deadband_drift_px: float = 2.0,
        row_corridors: Sequence[RowCorridorConfig] | None = None,
        min_car_width_px: float = 42.0,
        min_moto_width_px: float = 18.0,
    ) -> None:
        self.polygon_map = polygon_map
        self.tau_drift_px = tau_drift_px
        self.deadband_drift_px = deadband_drift_px
        self.min_car_width_px = min_car_width_px
        self.min_moto_width_px = min_moto_width_px

        self.anchors: list[GroundAnchor] = [
            GroundAnchor(
                anchor_id=a.get("id", f"anchor_{i}"),
                image_xy=(float(a["image_xy"][0]), float(a["image_xy"][1])),
                bev_xy=(float(a["bev_xy"][0]), float(a["bev_xy"][1])),
            )
            for i, a in enumerate(polygon_map.ground_anchors)
        ]

        self.row_corridors: list[RowCorridorConfig] = list(row_corridors or [])
        if not self.row_corridors and "row_corridors" in polygon_map.raw_config:
            for rc in polygon_map.raw_config["row_corridors"]:
                self.row_corridors.append(
                    RowCorridorConfig(
                        row_id=rc["row_id"],
                        space_ids=list(rc["space_ids"]),
                        compatibility=SpaceCompatibility(rc["compatibility"]),
                        quad_bev=[tuple(pt) for pt in rc["quad_bev"]],  # type: ignore[misc]
                        policy_mode=RowPolicyMode(rc.get("policy_mode", "strict_stalls")),
                        min_vehicle_width_px=float(rc.get("min_vehicle_width_px", min_car_width_px)),
                    )
                )

        # Precompute baseline depth distortion factor J_i for each registered slot
        self._baseline_j: dict[str, float] = self._compute_all_depth_factors(
            polygon_map.homography_matrix
        )

    # ------------------------------------------------------------------
    # 1. Anchor-Guided Micro-Drift Compensation (delta_H)
    # ------------------------------------------------------------------

    def estimate_drift_homography(
        self,
        observed_anchor_image_pts: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, float, bool, bool]:
        """
        Estimate effective homography H_eff = delta_H @ H_0 from observed ground anchors.

        Returns:
            (H_eff, mean_drift_bev_px, drift_adapted, camera_moved_fault)
        """
        H0 = self.polygon_map.homography_matrix
        if not observed_anchor_image_pts or len(self.anchors) < 4:
            return H0.copy(), 0.0, False, False

        matched_expected_bev: list[list[float]] = []
        matched_observed_bev: list[list[float]] = []

        for anchor in self.anchors:
            if anchor.anchor_id not in observed_anchor_image_pts:
                continue
            obs_ix, obs_iy = observed_anchor_image_pts[anchor.anchor_id]
            obs_bx, obs_by = self.polygon_map.project_point_to_bev(obs_ix, obs_iy, homography=H0)
            matched_expected_bev.append([anchor.bev_xy[0], anchor.bev_xy[1]])
            matched_observed_bev.append([obs_bx, obs_by])

        if len(matched_expected_bev) < 4:
            logger.warning("Insufficient visible anchors (%d < 4) — marking coverage fault.", len(matched_expected_bev))
            return H0.copy(), float("inf"), False, True

        exp_arr = np.array(matched_expected_bev, dtype=np.float64)
        obs_arr = np.array(matched_observed_bev, dtype=np.float64)
        drift_px = float(np.mean(np.linalg.norm(obs_arr - exp_arr, axis=1)))

        if drift_px <= self.deadband_drift_px:
            return H0.copy(), drift_px, False, False

        if drift_px > self.tau_drift_px:
            logger.error(
                "Camera drift %.2f px exceeds threshold %.2f px — refusing auto-adaptation.",
                drift_px,
                self.tau_drift_px,
            )
            return H0.copy(), drift_px, False, True

        delta_H, _ = cv2.findHomography(obs_arr.astype(np.float32), exp_arr.astype(np.float32), cv2.RANSAC, 3.0)
        if delta_H is None:
            return H0.copy(), drift_px, False, True

        H_eff = delta_H.astype(np.float64) @ H0
        return H_eff, drift_px, True, False

    # ------------------------------------------------------------------
    # 2. Row-Corridor Auto-Partitioning (Paint-Line Snap / Metric Fallback)
    # ------------------------------------------------------------------

    def partition_row_corridor(
        self,
        corridor: RowCorridorConfig,
        bev_image: np.ndarray | None = None,
    ) -> dict[str, ShapelyPolygon]:
        """
        Subdivide a parent BEV row quadrilateral into K ordered slot polygons
        matching corridor.space_ids. Snaps internal dividers to strong painted lines
        if bev_image is supplied; otherwise uses uniform metric partitioning.
        """
        k = len(corridor.space_ids)
        if k == 0:
            return {}

        tl, tr, br, bl = [np.array(pt, dtype=np.float64) for pt in corridor.quad_bev]
        # Default uniform fractions t in [0, 1]
        fractions = np.linspace(0.0, 1.0, k + 1)

        if bev_image is not None and k > 1:
            fractions = self._snap_dividers_to_paint(bev_image, tl, tr, br, bl, fractions)

        result: dict[str, ShapelyPolygon] = {}
        for idx, space_id in enumerate(corridor.space_ids):
            t0, t1 = float(fractions[idx]), float(fractions[idx + 1])
            p0 = tl + t0 * (tr - tl)
            p1 = tl + t1 * (tr - tl)
            p2 = bl + t1 * (br - bl)
            p3 = bl + t0 * (br - bl)
            poly = ShapelyPolygon([tuple(p0), tuple(p1), tuple(p2), tuple(p3)])
            result[space_id] = poly if poly.is_valid else poly.buffer(0)

        return result

    @staticmethod
    def _snap_dividers_to_paint(
        bev_image: np.ndarray,
        tl: np.ndarray,
        tr: np.ndarray,
        br: np.ndarray,
        bl: np.ndarray,
        uniform_fractions: np.ndarray,
        search_window_ratio: float = 0.15,
    ) -> np.ndarray:
        """Sample brightness gradient along the row axis to snap internal dividers to white/yellow paint."""
        gray = cv2.cvtColor(bev_image, cv2.COLOR_BGR2GRAY) if bev_image.ndim == 3 else bev_image
        h, w = gray.shape[:2]
        n_samples = 200
        ts = np.linspace(0.0, 1.0, n_samples)
        profile = np.zeros(n_samples, dtype=np.float64)

        for i, t in enumerate(ts):
            top_pt = tl + t * (tr - tl)
            bot_pt = bl + t * (br - bl)
            # Sample 15 points across bay depth
            alphas = np.linspace(0.15, 0.85, 15)
            vals = []
            for a in alphas:
                px = int(round(top_pt[0] + a * (bot_pt[0] - top_pt[0])))
                py = int(round(top_pt[1] + a * (bot_pt[1] - top_pt[1])))
                if 0 <= px < w and 0 <= py < h:
                    vals.append(float(gray[py, px]))
            profile[i] = float(np.mean(vals)) if vals else 0.0

        # High-pass response for painted stripes
        smooth = np.convolve(profile, np.ones(9) / 9.0, mode="same")
        paint_response = profile - smooth

        snapped = uniform_fractions.copy()
        step = 1.0 / (len(uniform_fractions) - 1)
        half_win = step * search_window_ratio

        for idx in range(1, len(snapped) - 1):
            center_t = uniform_fractions[idx]
            mask = (ts >= center_t - half_win) & (ts <= center_t + half_win)
            if np.any(mask):
                sub_ts = ts[mask]
                sub_resp = paint_response[mask]
                best_idx = int(np.argmax(sub_resp))
                if sub_resp[best_idx] > 12.0:  # Minimum contrast threshold for painted stripe
                    snapped[idx] = float(sub_ts[best_idx])

        return snapped

    # ------------------------------------------------------------------
    # 3. Occupancy-Guided Effective Boundary Adaptation (strict_stalls vs adaptive_curb)
    # ------------------------------------------------------------------

    def adapt_frame(
        self,
        frame: Frame,
        footprints: Sequence[ProjectedVehicleFootprint],
        observed_anchor_image_pts: dict[str, tuple[float, float]] | None = None,
    ) -> AdaptiveMapResult:
        """
        Run full adaptive mapping for a single frame:
          1. Check resolution and micro-drift via anchors.
          2. Partition any configured row corridors.
          3. Compute effective usable polygons P_eff per slot under strict_stalls or adaptive_curb.
          4. Attach perspective depth distortion factor J_i.
        """
        H_eff, drift_px, drift_adapted, moved_fault = self.estimate_drift_homography(
            observed_anchor_image_pts
        )

        # Resolution mismatch or excessive camera movement -> invalidate coverage
        if frame.resolution != self.polygon_map.reference_resolution or moved_fault:
            reason = "resolution_mismatch" if frame.resolution != self.polygon_map.reference_resolution else "camera_moved_fault"
            invalid_slots = [
                AdaptiveSlotGeometry(
                    space_id=sp.space_id,
                    slot_number=sp.slot_number,
                    lot_code=sp.lot_code,
                    compatibility=sp.compatibility,
                    policy_mode=RowPolicyMode.STRICT_STALLS,
                    baseline_polygon=sp.shapely_polygon,
                    effective_polygon=sp.shapely_polygon,
                    slot_area=sp.area_px,
                    effective_area=0.0,
                    usable_width_px=0.0,
                    depth_distortion_j=self._baseline_j.get(sp.space_id, 1.0),
                    is_valid_coverage=False,
                    drift_adapted=False,
                    status_reason=reason,
                )
                for sp in self.polygon_map.spaces
            ]
            return AdaptiveMapResult(
                calibration_version=self.polygon_map.calibration_version,
                camera_angle_type=self.polygon_map.camera_angle_type,
                effective_homography=H_eff,
                drift_px=drift_px,
                drift_adapted=False,
                camera_moved_fault=True,
                slots=invalid_slots,
            )

        # Partition any row corridors if configured
        bev_img = self.polygon_map.warp_to_bev(frame.image, homography=H_eff) if self.row_corridors else None
        corridor_polys: dict[str, ShapelyPolygon] = {}
        corridor_modes: dict[str, RowCorridorConfig] = {}
        for rc in self.row_corridors:
            sub_polys = self.partition_row_corridor(rc, bev_image=bev_img)
            corridor_polys.update(sub_polys)
            for sid in rc.space_ids:
                corridor_modes[sid] = rc

        # Evaluate adaptive_curb rows collectively first
        curb_slot_overrides: dict[str, tuple[ShapelyPolygon, float, bool]] = {}
        for rc in self.row_corridors:
            if rc.policy_mode == RowPolicyMode.ADAPTIVE_CURB:
                curb_slot_overrides.update(
                    self._evaluate_adaptive_curb_row(rc, corridor_polys, footprints)
                )

        j_factors = self._compute_all_depth_factors(H_eff) if drift_adapted else self._baseline_j
        vehicle_polys = [fp.footprint_bev for fp in footprints if not fp.footprint_bev.is_empty]

        adapted_slots: list[AdaptiveSlotGeometry] = []
        for sp in self.polygon_map.spaces:
            base_poly = corridor_polys.get(sp.space_id, sp.shapely_polygon)
            rc_cfg = corridor_modes.get(sp.space_id)
            mode = rc_cfg.policy_mode if rc_cfg else RowPolicyMode.STRICT_STALLS
            min_w = (
                rc_cfg.min_vehicle_width_px
                if rc_cfg
                else (self.min_car_width_px if sp.compatibility == SpaceCompatibility.CAR_ONLY else self.min_moto_width_px)
            )

            if sp.space_id in curb_slot_overrides:
                eff_poly, usable_w, blocked = curb_slot_overrides[sp.space_id]
                reason = "adaptive_curb_blocked" if blocked else "adaptive_curb_open"
            else:
                eff_poly, usable_w, blocked = self._evaluate_strict_stall(
                    base_poly, vehicle_polys, min_w
                )
                reason = "strict_stall_encroached" if blocked else "nominal"

            adapted_slots.append(
                AdaptiveSlotGeometry(
                    space_id=sp.space_id,
                    slot_number=sp.slot_number,
                    lot_code=sp.lot_code,
                    compatibility=sp.compatibility,
                    policy_mode=mode,
                    baseline_polygon=base_poly,
                    effective_polygon=eff_poly,
                    slot_area=float(base_poly.area),
                    effective_area=float(eff_poly.area),
                    usable_width_px=usable_w,
                    depth_distortion_j=j_factors.get(sp.space_id, 1.0),
                    is_valid_coverage=True,
                    drift_adapted=drift_adapted,
                    unlined_slot_blocked=blocked,
                    status_reason=reason,
                )
            )

        return AdaptiveMapResult(
            calibration_version=self.polygon_map.calibration_version,
            camera_angle_type=self.polygon_map.camera_angle_type,
            effective_homography=H_eff,
            drift_px=drift_px,
            drift_adapted=drift_adapted,
            camera_moved_fault=False,
            slots=adapted_slots,
        )

    @staticmethod
    def _evaluate_strict_stall(
        base_poly: ShapelyPolygon,
        vehicle_polys: Sequence[ShapelyPolygon],
        min_width_px: float,
    ) -> tuple[ShapelyPolygon, float, bool]:
        """Subtract overlapping vehicle footprints from a painted stall to find remaining usable geometry."""
        remaining = base_poly
        for vpoly in vehicle_polys:
            if remaining.intersects(vpoly):
                remaining = remaining.difference(vpoly)

        if remaining.is_empty:
            return ShapelyPolygon(), 0.0, True

        # Keep largest contiguous piece if fragmented
        if remaining.geom_type == "MultiPolygon":
            remaining = max(remaining.geoms, key=lambda g: g.area)

        minx, miny, maxx, maxy = remaining.bounds
        usable_w = float(min(maxx - minx, maxy - miny))
        blocked = usable_w < min_width_px or remaining.area < 0.55 * base_poly.area
        return remaining, usable_w, blocked

    @staticmethod
    def _evaluate_adaptive_curb_row(
        corridor: RowCorridorConfig,
        corridor_polys: dict[str, ShapelyPolygon],
        footprints: Sequence[ProjectedVehicleFootprint],
    ) -> dict[str, tuple[ShapelyPolygon, float, bool]]:
        """
        For unlined concrete rows (adaptive_curb), compute how many compatible vehicles
        can actually fit in the contiguous open intervals along the row, preventing
        fragmented half-gaps from being counted as available spaces.
        """
        tl, tr, br, bl = [np.array(pt, dtype=np.float64) for pt in corridor.quad_bev]
        row_vec = tr - tl
        row_len = float(np.linalg.norm(row_vec))
        if row_len < 1e-6:
            return {}
        u = row_vec / row_len
        row_poly = ShapelyPolygon([tuple(tl), tuple(tr), tuple(br), tuple(bl)])

        # Project occupied vehicle intervals onto 1D row axis [0, row_len]
        occupied_intervals: list[tuple[float, float]] = []
        for fp in footprints:
            inter = row_poly.intersection(fp.footprint_bev)
            if inter.is_empty or inter.area < 50.0:
                continue
            coords = list(inter.convex_hull.exterior.coords)
            projs = [float(np.dot(np.array(c) - tl, u)) for c in coords]
            s0 = max(0.0, min(projs))
            s1 = min(row_len, max(projs))
            if s1 > s0:
                occupied_intervals.append((s0, s1))

        occupied_intervals.sort()
        merged: list[list[float]] = []
        for s0, s1 in occupied_intervals:
            if not merged or s0 > merged[-1][1]:
                merged.append([s0, s1])
            else:
                merged[-1][1] = max(merged[-1][1], s1)

        # Find contiguous free intervals and count integer vehicle capacities
        free_intervals: list[tuple[float, float]] = []
        cursor = 0.0
        for s0, s1 in merged:
            if s0 > cursor:
                free_intervals.append((cursor, s0))
            cursor = max(cursor, s1)
        if cursor < row_len:
            free_intervals.append((cursor, row_len))

        total_open_capacity = sum(
            int((f1 - f0) // corridor.min_vehicle_width_px) for f0, f1 in free_intervals
        )
        k = len(corridor.space_ids)
        total_open_capacity = min(k, max(0, total_open_capacity))
        num_blocked = k - total_open_capacity

        # Assign blocked status deterministically to the first `num_blocked` registered IDs
        # and open status to the remaining `total_open_capacity` IDs
        out: dict[str, tuple[ShapelyPolygon, float, bool]] = {}
        for idx, sid in enumerate(corridor.space_ids):
            base = corridor_polys[sid]
            if idx < num_blocked:
                out[sid] = (ShapelyPolygon(), 0.0, True)
            else:
                out[sid] = (base, corridor.min_vehicle_width_px, False)
        return out

    # ------------------------------------------------------------------
    # 4. Perspective-Depth Distortion Factor (J_i via local Jacobian of H⁻¹)
    # ------------------------------------------------------------------

    def compute_depth_distortion_j(
        self,
        bev_xy: tuple[float, float],
        homography: np.ndarray | None = None,
        delta_bev: float = 5.0,
    ) -> float:
        """
        Compute local perspective stretch factor J_i >= 1.0 at BEV point (bx, by).
        Higher J_i means fewer raw camera pixels per unit ground area (e.g., far CCTV row),
        signaling wider spatial uncertainty to the downstream Fusion Model.
        """
        H = self.polygon_map.homography_matrix if homography is None else homography
        bx, by = bev_xy
        bev_quad = [
            (bx - delta_bev, by - delta_bev),
            (bx + delta_bev, by - delta_bev),
            (bx + delta_bev, by + delta_bev),
            (bx - delta_bev, by + delta_bev),
        ]
        img_quad = self.polygon_map.project_bev_poly_to_image(bev_quad, homography=H)
        img_area = ShapelyPolygon(img_quad).area
        bev_area = (2.0 * delta_bev) ** 2

        if img_area <= 1e-6:
            return 10.0

        # Ratio of BEV area to raw image pixel area (normalized so ~1.0 at 1:1 scale)
        stretch = bev_area / img_area
        return float(np.clip(stretch, 1.0, 10.0))

    def _compute_all_depth_factors(self, homography: np.ndarray) -> dict[str, float]:
        factors: dict[str, float] = {}
        for sp in self.polygon_map.spaces:
            c = sp.shapely_polygon.centroid
            factors[sp.space_id] = self.compute_depth_distortion_j(
                (float(c.x), float(c.y)), homography=homography
            )
        return factors