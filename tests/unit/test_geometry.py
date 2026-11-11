"""
tests/unit/test_geometry.py — Unit tests for geometry.py

Tests:
  - PolygonMap.from_json() validation (degenerate, duplicate IDs, mixed compat)
  - SpaceConfig polygon validation (too few points, out-of-bounds, tiny area)
  - project_point_to_bev() with identity homography
  - classify() — occupied, available, incompatible type, resolution mismatch → unknown
  - validate_polygon() edge cases
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from parking_occupancy.capture import CaptureConfig, Frame
from parking_occupancy.geometry import (
    PhysicalStatus,
    PolygonMap,
    SpaceCompatibility,
    SpaceConfig,
)
from parking_occupancy.inference import Detection, SyntheticDetector, VehicleClass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IDENTITY_H = np.eye(3, dtype=np.float64).tolist()

VALID_POLYGON_CONFIG = {
    "calibration_version": "test-v1",
    "camera_id": "cam-test",
    "lot_code": "A",
    "reference_resolution": [640, 480],
    "bev_canvas_size": [400, 300],
    "homography_matrix": IDENTITY_H,
    "src_points": [[0, 0], [640, 0], [640, 480], [0, 480]],
    "dst_points": [[0, 0], [400, 0], [400, 300], [0, 300]],
    "spaces": [
        {
            "space_id": "30000000-0000-4000-8000-000000000001",
            "lot_code": "A",
            "compatibility": "car_only",
            "polygon_bev": [[50, 50], [150, 50], [150, 150], [50, 150]],
        },
        {
            "space_id": "30000000-0000-4000-8000-000000000002",
            "lot_code": "A",
            "compatibility": "motorcycle_only",
            "polygon_bev": [[200, 50], [280, 50], [280, 120], [200, 120]],
        },
    ],
}


def write_config(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def make_frame(resolution=(640, 480)) -> Frame:
    w, h = resolution
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return Frame(image=img, timestamp=1.0, camera_id="cam-test", frame_index=0)


# ---------------------------------------------------------------------------
# PolygonMap.from_json — valid load
# ---------------------------------------------------------------------------

def test_load_valid_config():
    path = write_config(VALID_POLYGON_CONFIG)
    pmap = PolygonMap.from_json(path)
    assert pmap.camera_id == "cam-test"
    assert pmap.calibration_version == "test-v1"
    assert len(pmap.spaces) == 2
    assert pmap.homography_matrix.shape == (3, 3)


# ---------------------------------------------------------------------------
# PolygonMap.from_json — invalid configs
# ---------------------------------------------------------------------------

def test_load_missing_key():
    data = dict(VALID_POLYGON_CONFIG)
    del data["homography_matrix"]
    path = write_config(data)
    with pytest.raises(ValueError, match="homography_matrix"):
        PolygonMap.from_json(path)


def test_load_duplicate_space_id():
    data = dict(VALID_POLYGON_CONFIG)
    data["spaces"] = [
        VALID_POLYGON_CONFIG["spaces"][0],
        VALID_POLYGON_CONFIG["spaces"][0],  # duplicate
    ]
    path = write_config(data)
    with pytest.raises(ValueError, match="Duplicate"):
        PolygonMap.from_json(path)


def test_load_invalid_compatibility():
    data = dict(VALID_POLYGON_CONFIG)
    data["spaces"] = [
        {
            **VALID_POLYGON_CONFIG["spaces"][0],
            "compatibility": "mixed",  # not allowed
        }
    ]
    path = write_config(data)
    with pytest.raises(ValueError, match="mixed"):
        PolygonMap.from_json(path)


def test_load_homography_wrong_shape():
    data = dict(VALID_POLYGON_CONFIG)
    data["homography_matrix"] = [[1, 0], [0, 1]]  # 2×2, not 3×3
    path = write_config(data)
    with pytest.raises(ValueError, match="3×3"):
        PolygonMap.from_json(path)


# ---------------------------------------------------------------------------
# SpaceConfig — degenerate polygon rejection
# ---------------------------------------------------------------------------

def test_space_too_few_points():
    with pytest.raises(ValueError, match="3 points"):
        SpaceConfig(
            space_id="test-1",
            lot_code="A",
            compatibility=SpaceCompatibility.CAR_ONLY,
            polygon_bev=[[0, 0], [100, 0]],  # only 2 points
        )


def test_space_tiny_area():
    with pytest.raises(ValueError, match="degenerate"):
        SpaceConfig(
            space_id="test-2",
            lot_code="A",
            compatibility=SpaceCompatibility.CAR_ONLY,
            polygon_bev=[[0, 0], [1, 0], [0, 1]],  # area = 0.5 px²
        )


# ---------------------------------------------------------------------------
# project_point_to_bev — identity homography
# ---------------------------------------------------------------------------

def test_project_identity():
    path = write_config(VALID_POLYGON_CONFIG)
    pmap = PolygonMap.from_json(path)
    # With identity H, projection should be near-identity
    bx, by = pmap.project_point_to_bev(100.0, 200.0)
    assert abs(bx - 100.0) < 0.5
    assert abs(by - 200.0) < 0.5


# ---------------------------------------------------------------------------
# classify() — core occupancy logic
# ---------------------------------------------------------------------------

def _make_pmap() -> PolygonMap:
    path = write_config(VALID_POLYGON_CONFIG)
    return PolygonMap.from_json(path)


def test_classify_no_detections_gives_available():
    pmap = _make_pmap()
    frame = make_frame()
    states = pmap.classify(frame, [])
    # No detections → all tentatively available (before smoothing)
    for s in states:
        assert s.physical_status == PhysicalStatus.AVAILABLE


def test_classify_car_in_car_only_space_gives_occupied():
    pmap = _make_pmap()
    frame = make_frame()
    # Detection centroid at (100, 100) — inside space 1 polygon [50-150, 50-150]
    det = Detection(VehicleClass.CAR, 0.9, 80.0, 80.0, 120.0, 120.0)
    states = pmap.classify(frame, [det])
    space1 = next(s for s in states if s.space_id.endswith("000000000001"))
    assert space1.physical_status == PhysicalStatus.OCCUPIED
    assert space1.confidence is not None
    assert space1.confidence > 0.8


def test_classify_motorcycle_in_car_only_space_still_occupied():
    """Wrong vehicle type physically occupies the space — still OCCUPIED."""
    pmap = _make_pmap()
    frame = make_frame()
    # Motorcycle inside car_only space
    det = Detection(VehicleClass.MOTORCYCLE, 0.85, 80.0, 80.0, 120.0, 120.0)
    states = pmap.classify(frame, [det])
    space1 = next(s for s in states if s.space_id.endswith("000000000001"))
    assert space1.physical_status == PhysicalStatus.OCCUPIED


def test_classify_car_outside_polygons_gives_available():
    pmap = _make_pmap()
    frame = make_frame()
    # Car far outside both polygons
    det = Detection(VehicleClass.CAR, 0.9, 10.0, 10.0, 30.0, 30.0)
    states = pmap.classify(frame, [det])
    for s in states:
        assert s.physical_status == PhysicalStatus.AVAILABLE


def test_classify_resolution_mismatch_gives_unknown():
    """Frame resolution != reference_resolution → all unknown."""
    pmap = _make_pmap()
    # Create frame with wrong resolution
    frame = make_frame(resolution=(1920, 1080))
    states = pmap.classify(frame, [])
    for s in states:
        assert s.physical_status == PhysicalStatus.UNKNOWN
        assert s.confidence is None


# ---------------------------------------------------------------------------
# validate_polygon — static helper
# ---------------------------------------------------------------------------

def test_validate_polygon_ok():
    PolygonMap.validate_polygon(
        [(10, 10), (200, 10), (200, 200), (10, 200)],
        canvas_size=(400, 300),
    )


def test_validate_polygon_out_of_bounds():
    with pytest.raises(ValueError, match="out of canvas"):
        PolygonMap.validate_polygon(
            [(10, 10), (500, 10), (500, 200), (10, 200)],  # 500 > canvas width 400
            canvas_size=(400, 300),
        )


def test_validate_polygon_too_few_vertices():
    with pytest.raises(ValueError, match="3 vertices"):
        PolygonMap.validate_polygon([(10, 10), (200, 10)], canvas_size=(400, 300))
