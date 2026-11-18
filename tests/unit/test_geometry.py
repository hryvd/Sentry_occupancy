# ---------------------------------------------------------------------------
# Module 1 Extensions: Dual-source configs, footprint projection, H⁻¹ inverse
# ---------------------------------------------------------------------------
from pathlib import Path
from parking_occupancy.geometry import PolygonMap
from parking_occupancy.inference import Detection, VehicleClass


def test_load_dummy_dual_source_configs():
    cctv_map = PolygonMap.from_json(Path("configs/polygons/dummy_cctv.json"))
    opt_map = PolygonMap.from_json(Path("configs/polygons/dummy_optimized.json"))

    assert cctv_map.camera_angle_type.value == "cctv_oblique"
    assert opt_map.camera_angle_type.value == "optimized_angle"
    assert len(cctv_map.ground_anchors) == 4


def test_footprint_projection_cctv_vs_optimized():
    cctv_map = PolygonMap.from_json(Path("configs/polygons/dummy_cctv.json"))
    opt_map = PolygonMap.from_json(Path("configs/polygons/dummy_optimized.json"))

    # Tall bounding box (height = 100 px)
    det = Detection(VehicleClass.CAR, 0.90, 200.0, 200.0, 280.0, 300.0)

    fp_cctv = cctv_map.project_detection_footprint(det, ground_fraction=0.35)
    fp_opt = opt_map.project_detection_footprint(det)

    assert fp_cctv.footprint_bev.is_valid
    assert fp_cctv.footprint_bev.area > 0.0
    assert fp_opt.footprint_bev.is_valid
    assert fp_opt.footprint_bev.area > 0.0


def test_inverse_projection_roundtrip():
    cctv_map = PolygonMap.from_json(Path("configs/polygons/dummy_cctv.json"))
    orig_x, orig_y = 320.0, 280.0
    bx, by = cctv_map.project_point_to_bev(orig_x, orig_y)
    rx, ry = cctv_map.project_bev_point_to_image(bx, by)

    assert abs(rx - orig_x) < 1e-3
    assert abs(ry - orig_y) < 1e-3