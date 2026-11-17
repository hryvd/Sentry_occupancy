"""
tests/unit/test_adaptive_mapping.py — Unit tests for Module 1B (adaptive_mapping.py).

Covers:
  1. Micro-drift recovery via ground anchors (delta_H)
  2. Moved camera rejection when drift > tau_drift_px -> invalid coverage
  3. Unlined row partitioning (adaptive_curb) preventing fragmented gap double-counting
  4. Painted stall spillover (strict_stalls) shrinking effective_area & usable_width
  5. Far vs. near perspective depth distortion factor (J_i) on oblique CCTV
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from parking_occupancy.adaptive_mapping import (
    AdaptiveSlotMapper,
    RowCorridorConfig,
    RowPolicyMode,
)
from parking_occupancy.capture import Frame
from parking_occupancy.geometry import (
    PolygonMap,
    ProjectedVehicleFootprint,
    SpaceCompatibility,
)
from parking_occupancy.inference import Detection, VehicleClass


def _make_frame(resolution=(640, 480), camera_id="cam-dummy-cctv-01") -> Frame:
    w, h = resolution
    return Frame(
        image=np.full((h, w, 3), 180, dtype=np.uint8),
        timestamp=10.0,
        camera_id=camera_id,
        frame_index=1,
    )


def test_micro_drift_recovery():
    """Scenario 1: Small 5 px anchor shift is recovered via delta_H."""
    pmap = PolygonMap.from_json(Path("configs/polygons/dummy_optimized.json"))
    mapper = AdaptiveSlotMapper(pmap, tau_drift_px=12.0, deadband_drift_px=1.0)

    # Shift all anchors by +8 px in image space (+5 px in BEV space)
    shifted_anchors = {
        a["id"]: (a["image_xy"][0] + 8.0, a["image_xy"][1] + 8.0)
        for a in pmap.ground_anchors
    }

    res = mapper.adapt_frame(_make_frame(camera_id=pmap.camera_id), [], observed_anchor_image_pts=shifted_anchors)
    assert res.drift_adapted is True
    assert res.camera_moved_fault is False
    assert 4.0 < res.drift_px < 8.0
    assert all(s.is_valid_coverage for s in res.slots)


def test_moved_camera_rejection():
    """Scenario 2: Large shift (> tau_drift_px) refuses adaptation and flags fault."""
    pmap = PolygonMap.from_json(Path("configs/polygons/dummy_optimized.json"))
    mapper = AdaptiveSlotMapper(pmap, tau_drift_px=10.0)

    # Shift anchors by +80 px in image space (~50 px in BEV > 10 px limit)
    shifted_anchors = {
        a["id"]: (a["image_xy"][0] + 80.0, a["image_xy"][1])
        for a in pmap.ground_anchors
    }

    res = mapper.adapt_frame(_make_frame(camera_id=pmap.camera_id), [], observed_anchor_image_pts=shifted_anchors)
    assert res.camera_moved_fault is True
    assert res.drift_adapted is False
    assert all(not s.is_valid_coverage for s in res.slots)


def test_unlined_row_adaptive_curb_partitioning():
    """Scenario 3: 2 poorly spaced cars in a 4-car unlined strip leave fragmented gaps."""
    pmap = PolygonMap.from_json(Path("configs/polygons/dummy_optimized.json"))
    corridor = RowCorridorConfig(
        row_id="row-unlined-1",
        space_ids=[
            "30000000-0000-4000-8000-000000000101",
            "30000000-0000-4000-8000-000000000102",
        ],
        compatibility=SpaceCompatibility.CAR_ONLY,
        quad_bev=[(40.0, 40.0), (240.0, 40.0), (240.0, 140.0), (40.0, 140.0)],
        policy_mode=RowPolicyMode.ADAPTIVE_CURB,
        min_vehicle_width_px=90.0,
    )
    mapper = AdaptiveSlotMapper(pmap, row_corridors=[corridor])

    # Park 1 car right in the middle [95..185], leaving [40..95] (55px) and [185..240] (55px)
    # Neither side gap fits a 90px car -> 0 open capacity (both slots blocked)
    mid_car = ProjectedVehicleFootprint(
        detection=Detection(VehicleClass.CAR, 0.9, 0, 0, 10, 10),
        footprint_bev=ShapelyPolygon([(95, 50), (185, 50), (185, 130), (95, 130)]),
        centroid_bev=(140.0, 90.0),
    )

    res = mapper.adapt_frame(_make_frame(camera_id=pmap.camera_id), [mid_car])
    assert all(s.unlined_slot_blocked for s in res.slots)


def test_painted_stall_spillover_strict_stalls():
    """Scenario 4: Car encroaching from B-C01 into B-C02 shrinks B-C02 usable width."""
    pmap = PolygonMap.from_json(Path("configs/polygons/dummy_optimized.json"))
    mapper = AdaptiveSlotMapper(pmap, min_car_width_px=80.0)

    # Car covers B-C01 [40..140] and spills 45px into B-C02 [150..195]
    spill_car = ProjectedVehicleFootprint(
        detection=Detection(VehicleClass.CAR, 0.9, 0, 0, 10, 10),
        footprint_bev=ShapelyPolygon([(60, 45), (195, 45), (195, 135), (60, 135)]),
        centroid_bev=(127.5, 90.0),
    )

    res = mapper.adapt_frame(_make_frame(camera_id=pmap.camera_id), [spill_car])
    slot2 = next(s for s in res.slots if s.slot_number == "B-C02")
    assert slot2.usable_width_px < 80.0
    assert slot2.unlined_slot_blocked is True


def test_far_vs_near_depth_distortion_j():
    """Scenario 5: Far row in oblique CCTV has higher J_i stretch than near row."""
    cctv_map = PolygonMap.from_json(Path("configs/polygons/dummy_cctv.json"))
    mapper = AdaptiveSlotMapper(cctv_map)

    res = mapper.adapt_frame(_make_frame(camera_id=cctv_map.camera_id), [])
    far_slot = next(s for s in res.slots if s.slot_number == "A-C01")   # y in [40..140] (far)
    near_slot = next(s for s in res.slots if s.slot_number == "A-C03")  # y in [160..260] (near)

    assert far_slot.depth_distortion_j > near_slot.depth_distortion_j