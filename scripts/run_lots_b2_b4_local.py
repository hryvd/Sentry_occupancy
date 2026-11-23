#!/usr/bin/env python3
"""
scripts/run_lots_b2_b4_local.py
===============================
Main optimized local runner & HTTP server for Lot B2 (22 bays) & Lot B4 (27 bays).

Runs the 4-model fusion pipeline:
  Model 1 (15%): Geometry (tyre-line footprint + MiDaS elevation)
  Model 2 (15%): Contextual slot analysis (free gap width, spillover, occlusion)
  Model 3 (20%): YOLOX-L (cached ONNX session + far-end zoom)
  Model 4 (50%): Top-down full geometry + context + YOLO re-analysis (highest weight)

Outputs:
  - CSV: area, slot_id, zone, compatibility, space_id, initial_state, model_1, model_2, model_3, model_4, mean, final_state, final_id, backend_display_code
  - 2D Maps: lot_b2_2d_map.png, lot_b4_2d_map.png, lot_b2_b4_combined_2d_map.png
  - SENTRY Mock Observations: 49 occupancy_observation events (correlation_id=null)
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import shutil
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
LOTS_DIR = Path(__file__).resolve().parent / "lots_b2_b4"
AREA_B2_DIR = Path(__file__).resolve().parent / "area_b2"
for _p in (str(SRC_DIR), str(LOTS_DIR), str(AREA_B2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fusion_4model as F4  # noqa: E402
import parking_lot2 as P2  # noqa: E402
import slot_occupancy as SO  # noqa: E402
import topview_slots as T  # noqa: E402

from parking_occupancy.backend_client import BackendClient, BackendMode  # noqa: E402
from parking_occupancy.capture import CaptureConfig, Frame  # noqa: E402
from parking_occupancy.events import EventEmitter  # noqa: E402
from parking_occupancy.freshness import FreshnessTracker  # noqa: E402
from parking_occupancy.geometry import PhysicalStatus, PolygonMap, SpaceState  # noqa: E402
from parking_occupancy.smoothing import StateSmoother  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lots_b2_b4_local")

REF_W, REF_H = 2048, 1536
_SERVER_LOCK = threading.Lock()
_B2_REF_SLOTS_CACHE: list[SO.Slot] | None = None


class _RecordingMockClient:
    """Delegate over BackendClient(BackendMode.MOCK) that captures emitted payloads."""

    def __init__(self, inner: BackendClient) -> None:
        self._inner = inner
        self.events: list[dict[str, Any]] = []

    @property
    def boot_id(self) -> str:
        return self._inner.boot_id

    def next_sequence(self, space_id: str) -> int:
        return self._inner.next_sequence(space_id)

    def post_observation(self, payload: dict[str, Any]) -> None:
        self.events.append(payload)
        self._inner.post_observation(payload)

    def post_heartbeat(self, payload: dict[str, Any]) -> None:
        self._inner.post_heartbeat(payload)

    def flush(self) -> None:
        self._inner.flush()

    def close(self) -> None:
        self._inner.close()


def _get_b2_ref_slots() -> list[SO.Slot]:
    """Build calibrated 22 Lot B2 reference bays once for fixed-slot custom ingestion."""
    global _B2_REF_SLOTS_CACHE
    if _B2_REF_SLOTS_CACHE is None:
        ref_path = ROOT_DIR / "ui/samples/area_b2_raw.jpg"
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            ref_img = np.zeros((REF_H, REF_W, 3), dtype=np.uint8)
        _, slots, _, _ = SO.run_lot1(ref_img)
        _B2_REF_SLOTS_CACHE = slots
    return _B2_REF_SLOTS_CACHE


def _emit_sentry_events(config_path: Path, scored_rows: list[dict[str, Any]], dummy_img: np.ndarray) -> list[dict[str, Any]]:
    """Emit SENTRY occupancy_observation events strictly in mock mode (correlation_id=null)."""
    cfg = CaptureConfig.from_yaml(config_path)
    poly_path = cfg.polygon_config if cfg.polygon_config.is_absolute() else ROOT_DIR / cfg.polygon_config
    polygon_map = PolygonMap.from_json(poly_path)
    spaces_meta = json.loads(poly_path.read_text(encoding="utf-8"))["spaces"]
    uuid_by_slot = {s["slot_number"]: s["space_id"] for s in spaces_meta}

    smoother = StateSmoother(n_to_occupied=1, n_to_available=1)
    freshness = FreshnessTracker.from_config(cfg)
    rec_client = _RecordingMockClient(BackendClient(mode=BackendMode.MOCK, config=cfg))
    emitter = EventEmitter(
        producer_id=cfg.device_id,
        model_version=cfg.model_version,
        calibration_version=polygon_map.calibration_version,
        client=rec_client,
    )

    frame = Frame(image=dummy_img, timestamp=time.monotonic(), camera_id=cfg.camera_id, frame_index=0)
    raw_states: list[SpaceState] = []
    space_by_id = {sp.space_id: sp for sp in polygon_map.spaces}
    for r in scored_rows:
        sp_id = uuid_by_slot[r["slot_id"]]
        sp = space_by_id[sp_id]
        st_enum = PhysicalStatus(r["final_state"])
        conf = None if st_enum == PhysicalStatus.UNKNOWN else round(float(r["mean"]), 4)
        raw_states.append(
            SpaceState(
                space_id=sp_id,
                lot_code=sp.lot_code,
                compatibility=sp.compatibility,
                physical_status=st_enum,
                confidence=conf,
            )
        )

    smoothed = smoother.update(frame.timestamp, raw_states)
    final_states = freshness.apply(frame.timestamp, smoothed)
    emitter.emit_changes(frame.timestamp, final_states)
    emitter.flush()
    rec_client.close()
    return rec_client.events


def _render_check_image(img_bgr: np.ndarray, tv, slots: list[SO.Slot], scored_rows: list[dict[str, Any]], yolox_boxes: np.ndarray | None) -> np.ndarray:
    """Render photo annotation showing YOLOX boxes + 4-model final bay badges."""
    vis = img_bgr.copy()
    to_photo = SO.photo_px(tv)
    row_by_id = {r["slot_id"]: r for r in scored_rows}

    if yolox_boxes is not None:
        for b in yolox_boxes:
            x0, y0, x1, y1, sc = int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(b[4])
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 195, 255), 2)
            cv2.putText(vis, f"{sc:.2f}", (x0 + 2, max(16, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

    for s in slots:
        r = row_by_id[s.id]
        cx, cy = to_photo(s.center[None])[0]
        if not (0 <= cx < vis.shape[1] and 0 <= cy < vis.shape[0]):
            continue
        col = (60, 62, 224) if r["final_id"] == 1 else (60, 175, 72)
        rad = 18 if s.row != "moto" else 13
        cv2.circle(vis, (int(cx), int(cy)), rad + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, (int(cx), int(cy)), rad, col, -1, cv2.LINE_AA)
        lbl = f"{s.id}"
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
        cv2.putText(vis, lbl, (int(cx) - tw // 2, int(cy) + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _render_b4_perspective_blueprint(tv, slots: list[SO.Slot], vehs: list[SO.Vehicle], scored_rows: list[dict[str, Any]]) -> np.ndarray:
    """Render a clean 2048x1536 calibrated camera-plane blueprint for Lot B4 when raw photo is absent."""
    canvas = np.full((REF_H, REF_W, 3), (24, 18, 12), dtype=np.uint8)
    to_photo = SO.photo_px(tv)
    row_by_id = {r["slot_id"]: r for r in scored_rows}

    # Subtle horizon & ground mesh
    for gy in np.linspace(3.0, 29.0, 14):
        pts = to_photo([[1.5, gy], [21.0, gy]]).astype(np.int32)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (42, 34, 24), 1, cv2.LINE_AA)
    for gx in np.linspace(2.0, 20.0, 10):
        pts = to_photo([[gx, 3.0], [gx, 29.0]]).astype(np.int32)
        cv2.line(canvas, tuple(pts[0]), tuple(pts[1]), (42, 34, 24), 1, cv2.LINE_AA)

    for s in slots:
        r = row_by_id[s.id]
        col = (60, 62, 224) if r["final_id"] == 1 else (65, 185, 78)
        pts = np.int32(np.round(to_photo(s.poly)))
        F4._draw_poly_alpha(canvas, pts, col, 0.25, outline=2)

    for v in vehs:
        x0, y0, x1, y1 = [int(round(q)) for q in v.box]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(canvas, f"{v.id}:{v.kind}", (x0 + 4, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)

    for s in slots:
        r = row_by_id[s.id]
        cx, cy = to_photo(s.center[None])[0]
        F4._draw_pill_label(canvas, f"{s.id} [{r['final_id']}]", (cx, cy), scale=0.45 if s.row != "moto" else 0.35)

    cv2.rectangle(canvas, (28, 24), (820, 108), (15, 23, 42), -1)
    cv2.rectangle(canvas, (28, 24), (820, 108), (56, 189, 248), 2)
    cv2.putText(canvas, "LOT B4 (27 BAYS: 18 CARS + 9 MOTO) - CALIBRATED PERSPECTIVE", (46, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 246, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Drop any Lot B4 camera image to run live YOLOX-L + MiDaS-small", (46, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 220, 255), 1, cv2.LINE_AA)
    return canvas


def _render_b4_topview_blueprint(tv, slots: list[SO.Slot], vehs: list[SO.Vehicle], scored_rows: list[dict[str, Any]]) -> np.ndarray:
    """Render metric rectified topview blueprint (1380x1800) for Lot B4 when raw photo is absent."""
    canvas = np.full((tv.out_h, tv.out_w, 3), (24, 24, 24), dtype=np.uint8)
    row_by_id = {r["slot_id"]: r for r in scored_rows}
    for s in slots:
        r = row_by_id[s.id]
        col = (60, 62, 224) if r["final_id"] == 1 else (65, 185, 78)
        F4._draw_poly_alpha(canvas, s.poly * SO.PPM, col, 0.30, outline=2)
    for v in vehs:
        F4._draw_poly_alpha(canvas, v.foot * SO.PPM, (165, 175, 190), 0.35, outline=2)
        cx, cy = v.center * SO.PPM
        F4._draw_pill_label(canvas, v.id, (cx, cy), scale=0.40, fg=(255, 255, 255), bg=(55, 65, 81))
    T.scale_bar(canvas, SO.PPM, margin=48)
    return canvas


def analyze_lot(
    area_code: str,
    img_bgr: np.ndarray | None,
    out_dir: Path,
    models_dir: Path,
    prefix: str = "lot_",
    is_custom: bool = False,
) -> dict[str, Any]:
    """Run full 4-model pipeline + 2D map + SENTRY mock emission for Lot B2 or Lot B4."""
    area_upper = area_code.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_res = "2048×1536 (calibrated)"

    if img_bgr is not None:
        oh, ow = img_bgr.shape[:2]
        orig_res = f"{ow}×{oh}"
        img_ref = cv2.resize(img_bgr, (REF_W, REF_H), interpolation=cv2.INTER_LINEAR) if (ow, oh) != (REF_W, REF_H) else img_bgr
        kept, live_typed, zoom_arr = F4.detect_vehicles_cached(img_ref, models_dir)
        z_at, heightmap, _ = F4.compute_midas_residuals(img_ref, models_dir)
    else:
        img_ref = None
        kept, live_typed, zoom_arr = np.empty((0, 6)), None, None
        z_at, heightmap = None, None

    if area_upper == "B2":
        if img_ref is None:
            raise ValueError("Lot B2 requires an image frame.")
        fixed_slots = _get_b2_ref_slots() if is_custom else None
        tv, slots, vehs, valid = SO.run_lot1(img_ref, live_boxes=live_typed, fixed_slots=fixed_slots, enable_stack=True)
        R, held, _ = SO.assign(slots, vehs, getattr(tv, "stacks", ()))
        to_photo = SO.photo_px(tv)
        rows_raw = SO.slot_status(
            slots, vehs, R, held, valid, to_photo, img_ref.shape,
            confirmed_free=() if is_custom else SO.LOT1_CONFIRMED_FREE,
            site_report=False if is_custom else SO.LOT1_SITE_REPORT,
        )
        top_img = T.render_topview(img_ref, tv)
        top_plain = top_img.copy()
        T.scale_bar(top_plain, SO.PPM)
        scored = F4.score_lot_4model(
            "B2", tv, slots, vehs, rows_raw, R,
            img_bgr=img_ref, topview_img=top_img, z_at=z_at, live_yolox_boxes=zoom_arr,
        )
        cfg_yaml = ROOT_DIR / "configs/cameras/area_b2.yaml"
        area_title = f"LOT B2 - 22 BAYS (L1-L11, R1-R11){' [CUSTOM]' if is_custom else ''}"
    elif area_upper == "B4":
        tv, slots, vehs, valid = SO.run_lot2(img_ref if img_ref is not None else (REF_H, REF_W), live_boxes=live_typed)
        R, held, _ = SO.assign(slots, vehs)
        to_photo = SO.photo_px(tv) if img_ref is not None else None
        rows_raw = SO.slot_status(
            slots, vehs, R, held, valid,
            to_photo=to_photo, shape=img_ref.shape if img_ref is not None else None,
        )
        if img_ref is not None:
            top_img = T.render_topview(img_ref, tv)
            top_plain = top_img.copy()
            T.scale_bar(top_plain, SO.PPM)
        else:
            top_img = None
            top_plain = None
        scored = F4.score_lot_4model(
            "B4", tv, slots, vehs, rows_raw, R,
            img_bgr=img_ref, topview_img=top_img, z_at=z_at, live_yolox_boxes=zoom_arr,
        )
        cfg_yaml = ROOT_DIR / "configs/cameras/area_b4.yaml"
        area_title = f"LOT B4 - 27 BAYS (18 CARS + 9 MOTO){' [CUSTOM]' if is_custom else ''}"
    else:
        raise ValueError(f"Unsupported lot area '{area_code}'. Expected 'B2' or 'B4'.")

    # Render outputs
    map_2d = F4.render_lot_2d_map(area_title, slots, vehs, valid, scored, tv.size_m)
    map_name = f"{prefix}2d_map.png"
    cv2.imwrite(str(out_dir / map_name), map_2d)

    if img_ref is not None:
        check_img = _render_check_image(img_ref, tv, slots, scored, kept[:, :5] if len(kept) else None)
        photo_ov = F4.render_photo_overlay(img_ref, tv, slots, vehs, scored)
        top_out = top_plain
        hm_out = heightmap
    else:
        # Clean calibrated blueprint visuals for B4 when no raw frame is present
        photo_ov = _render_b4_perspective_blueprint(tv, slots, vehs, scored)
        top_out = _render_b4_topview_blueprint(tv, slots, vehs, scored)
        check_img = photo_ov.copy()
        hm_out = photo_ov.copy()

    check_name = f"{prefix}check.jpg"
    hm_name = f"{prefix}heightmap.jpg"
    top_name = f"{prefix}topview.jpg"
    ov_name = f"{prefix}photo_overlay.jpg"
    cv2.imwrite(str(out_dir / check_name), check_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / hm_name), hm_out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / top_name), top_out, [cv2.IMWRITE_JPEG_QUALITY, 94])
    cv2.imwrite(str(out_dir / ov_name), photo_ov, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Emit SENTRY mock events
    dummy_frame_img = img_ref if img_ref is not None else photo_ov
    events = _emit_sentry_events(cfg_yaml, scored, dummy_frame_img)

    # Project anchor/topview coords cleanly for interactive UI display
    poly_path = CaptureConfig.from_yaml(cfg_yaml).polygon_config
    if not poly_path.is_absolute():
        poly_path = ROOT_DIR / poly_path
    spaces_meta = {s["slot_number"]: s for s in json.loads(poly_path.read_text(encoding="utf-8"))["spaces"]}
    to_photo_fn = SO.photo_px(tv)
    top_px_scale = 1.0
    for r in scored:
        cx_m, cy_m = r["center_m"]
        ax, ay = to_photo_fn([[cx_m, cy_m]])[0]
        r["anchor"] = [round(float(np.clip(ax, 0, REF_W)), 1), round(float(np.clip(ay, 0, REF_H)), 1)]
        r["topview"] = [round(float(cx_m * SO.PPM * top_px_scale), 1), round(float(cy_m * SO.PPM * top_px_scale), 1)]
        sp_meta = spaces_meta[r["slot_id"]]
        r["zone"] = sp_meta.get("zone", r.get("row", ""))
        r["compatibility"] = sp_meta.get("compatibility", r.get("compatibility", ""))
        r["space_id"] = sp_meta["space_id"]
        r["backend_display_code"] = sp_meta.get("backend_display_code", "TBD (B-P...)")
        poly_px_cam = to_photo_fn(r["poly_m"])
        r["cam_poly"] = [[round(float(x), 1), round(float(y), 1)] for x, y in poly_px_cam]

    ts_ms = int(time.time() * 1000)
    occ_cnt = sum(r["final_id"] == 1 for r in scored)
    tot_cnt = len(scored)
    return {
        "area": area_upper,
        "title": area_title,
        "original_res": orig_res,
        "area_m": [round(float(tv.size_m[0]), 1), round(float(tv.size_m[1]), 1)],
        "topview_size": [int(tv.out_w), int(tv.out_h)],
        "occupied_count": occ_cnt,
        "available_count": sum(r["final_state"] == "available" for r in scored),
        "unknown_count": sum(r["final_state"] == "unknown" for r in scored),
        "total_slots": tot_cnt,
        "occupancy_rate": round(occ_cnt / tot_cnt * 100.0, 1) if tot_cnt else 0.0,
        "weights": {"model_1": F4.WEIGHTS[0], "model_2": F4.WEIGHTS[1], "model_3": F4.WEIGHTS[2], "model_4": F4.WEIGHTS[3]},
        "slots": scored,
        "detected_boxes": [[round(float(x), 1) for x in b[:5]] for b in kept] if len(kept) else [],
        "emitted_events_count": len(events),
        "sample_event": events[0] if events else None,
        "map_img": map_2d,
        "overlay_urls": {
            "check": f"/outputs/lots_b2_b4/{check_name}?t={ts_ms}",
            "heightmap": f"/outputs/lots_b2_b4/{hm_name}?t={ts_ms}",
            "topview": f"/outputs/lots_b2_b4/{top_name}?t={ts_ms}",
            "photo_overlay": f"/outputs/lots_b2_b4/{ov_name}?t={ts_ms}",
            "plan": f"/outputs/lots_b2_b4/{map_name}?t={ts_ms}",
        },
        "files": {
            "2d_map": str(out_dir / map_name),
            "check": str(out_dir / check_name),
            "heightmap": str(out_dir / hm_name),
            "topview": str(out_dir / top_name),
            "photo_overlay": str(out_dir / ov_name),
        },
    }


def run_both_lots(
    b2_image: Path,
    b4_image: Path | None,
    out_dir: Path,
    models_dir: Path,
    sync_ui_samples: bool = True,
) -> dict[str, Any]:
    """Run 4-model pipeline on Lot B2 + Lot B4, generate combined 2D map, CSVs & JSON report."""
    t0 = time.perf_counter()
    img_b2 = cv2.imread(str(b2_image))
    if img_b2 is None:
        raise FileNotFoundError(f"Cannot read Lot B2 image: {b2_image}")
    img_b4 = None
    if b4_image is not None:
        img_b4 = cv2.imread(str(b4_image))
        if img_b4 is None:
            raise FileNotFoundError(f"Cannot read Lot B4 image: {b4_image}")

    b2_res = analyze_lot("B2", img_b2, out_dir, models_dir, prefix="lot_b2_", is_custom=False)
    b4_res = analyze_lot("B4", img_b4, out_dir, models_dir, prefix="lot_b4_", is_custom=b4_image is not None)

    b2_map = b2_res.pop("map_img")
    b4_map = b4_res.pop("map_img")
    all_rows = b2_res["slots"] + b4_res["slots"]
    tot_occ = sum(r["final_id"] == 1 for r in all_rows)
    tot_avail = sum(r["final_id"] == 0 for r in all_rows)
    tot_all = len(all_rows)
    summary_txt = (
        f"SENTRY LOT B2 & B4 - 2D OCCUPANCY MAP  |  "
        f"{tot_occ}/{tot_all} OCCUPIED ({tot_avail} AVAILABLE)  |  "
        f"M1(15%)+M2(15%)+M3(20%)+M4_TopDown(50%)"
    )
    combined_map = F4.render_combined_2d_map(b2_map, b4_map, summary_txt)
    combined_map_path = out_dir / "lot_b2_b4_combined_2d_map.png"
    cv2.imwrite(str(combined_map_path), combined_map)

    F4.write_occupancy_csv(b2_res["slots"], out_dir / "lot_b2_occupancy.csv")
    F4.write_occupancy_csv(b4_res["slots"], out_dir / "lot_b4_occupancy.csv")
    F4.write_occupancy_csv(all_rows, out_dir / "lot_b2_b4_occupancy.csv")

    if sync_ui_samples:
        ui_samples = ROOT_DIR / "ui/samples"
        ui_samples.mkdir(parents=True, exist_ok=True)
        for fname in (
            "lot_b2_2d_map.png",
            "lot_b4_2d_map.png",
            "lot_b2_b4_combined_2d_map.png",
            "lot_b2_topview.jpg",
            "lot_b2_photo_overlay.jpg",
            "lot_b2_check.jpg",
            "lot_b2_heightmap.jpg",
            "lot_b4_2d_map.png",
            "lot_b4_topview.jpg",
            "lot_b4_photo_overlay.jpg",
            "lot_b4_check.jpg",
            "lot_b4_heightmap.jpg",
        ):
            src_f = out_dir / fname
            if src_f.exists():
                shutil.copy2(src_f, ui_samples / fname)
        # Keep stable ui/samples filenames populated cleanly
        shutil.copy2(out_dir / "lot_b2_2d_map.png", ui_samples / "area_b2_plan.png")
        shutil.copy2(out_dir / "lot_b2_topview.jpg", ui_samples / "area_b2_topview.jpg")
        shutil.copy2(out_dir / "lot_b2_check.jpg", ui_samples / "area_b2_check.jpg")
        shutil.copy2(out_dir / "lot_b2_heightmap.jpg", ui_samples / "area_b2_heightmap.jpg")
        shutil.copy2(out_dir / "lot_b4_photo_overlay.jpg", ui_samples / "lot_b4_perspective.png")

    elapsed_s = round(time.perf_counter() - t0, 2)
    report = {
        "ok": True,
        "elapsed_s": elapsed_s,
        "weights": {"model_1": F4.WEIGHTS[0], "model_2": F4.WEIGHTS[1], "model_3": F4.WEIGHTS[2], "model_4": F4.WEIGHTS[3]},
        "totals": {
            "total_slots": tot_all,
            "occupied": tot_occ,
            "available": tot_avail,
            "occupancy_rate": round(tot_occ / tot_all * 100.0, 1) if tot_all else 0.0,
            "emitted_events_count": b2_res["emitted_events_count"] + b4_res["emitted_events_count"],
        },
        "lots": {"B2": b2_res, "B4": b4_res},
        "csv_path": str(out_dir / "lot_b2_b4_occupancy.csv"),
        "combined_2d_map_path": str(combined_map_path),
    }
    (out_dir / "sentry_b2_b4_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print unified 49-row 4-model audit table and paths."""
    tot = report["totals"]
    print("\n================ SENTRY LOT B2 & B4 — 4-MODEL FUSION AUDIT ================")
    print(f"Elapsed     : {report['elapsed_s']}s  (ONNX cached session)")
    print(f"Weights     : M1(Geom)=0.15  M2(Ctx)=0.15  M3(YOLO)=0.20  M4(TopDown)=0.50")
    print(
        f"Summary     : Total={tot['occupied']}/{tot['total_slots']} Occupied ({tot['occupancy_rate']}%) | "
        f"B2={report['lots']['B2']['occupied_count']}/22 | "
        f"B4={report['lots']['B4']['occupied_count']}/27 | "
        f"MockEvents={tot['emitted_events_count']} (correlation_id=null)"
    )
    print("-" * 104)
    print(",".join(F4.CSV_HEADER))
    for area_key in ("B2", "B4"):
        for r in report["lots"][area_key]["slots"]:
            print(
                f"{r['area']},{r['slot_id']},{r.get('zone', r.get('row', ''))},"
                f"{r.get('compatibility', '')},{r.get('space_id', '')},{r['initial_state']},"
                f"{r['model_1']:.4f},{r['model_2']:.4f},{r['model_3']:.4f},{r['model_4']:.4f},"
                f"{r['mean']:.4f},{r['final_state']},{r['final_id']},{r.get('backend_display_code', 'TBD (B-P...)')}"
            )
    print("-" * 104)
    print(f"CSV Output  : {report['csv_path']}")
    print(f"2D Map (B2) : {report['lots']['B2']['files']['2d_map']}")
    print(f"2D Map (B4) : {report['lots']['B4']['files']['2d_map']}")
    print(f"2D Combined : {report['combined_2d_map_path']}")
    print("===========================================================================\n")


def _serve_http(host: str, port: int, out_dir: Path, models_dir: Path, latest_report: dict[str, Any]) -> None:
    """Serve UI, /outputs/, /api/lots, /api/csv, and POST /api/analyze on localhost."""
    ui_dir = (ROOT_DIR / "ui").resolve()
    outputs_root = (ROOT_DIR / "outputs").resolve()
    state_holder = {"report": latest_report}

    class _Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, status: int, data: bytes, ctype: str, extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for hk, hv in extra_headers.items():
                    self.send_header(hk, hv)
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            self._send_bytes(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def _send_file(self, base_dir: Path, rel_path: str) -> None:
            clean_rel = urllib.parse.unquote(rel_path).lstrip("/") or "index.html"
            candidate = (base_dir / clean_rel).resolve()
            if base_dir not in candidate.parents and candidate != base_dir:
                self.send_error(403, "Forbidden")
                return
            if not candidate.is_file():
                self.send_error(404, "Not Found")
                return
            ctype, _ = mimetypes.guess_type(str(candidate))
            self._send_bytes(200, candidate.read_bytes(), ctype or "application/octet-stream")

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(200, {"ok": True, "mode": "lots_b2_b4_4model"})
                return
            if parsed.path == "/api/lots":
                self._send_json(200, state_holder["report"])
                return
            if parsed.path == "/api/csv":
                csv_file = out_dir / "lot_b2_b4_occupancy.csv"
                if not csv_file.is_file():
                    self.send_error(404, "CSV not found")
                    return
                self._send_bytes(
                    200,
                    csv_file.read_bytes(),
                    "text/csv; charset=utf-8",
                    {"Content-Disposition": 'attachment; filename="lot_b2_b4_occupancy.csv"'},
                )
                return
            if parsed.path.startswith("/outputs/"):
                self._send_file(outputs_root, parsed.path[len("/outputs/"):])
                return
            self._send_file(ui_dir, parsed.path)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != "/api/analyze":
                self.send_error(404, "Not Found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 32 * 1024 * 1024:
                    self._send_json(400, {"ok": False, "error": "Invalid body length."})
                    return
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                area = str(body.get("area") or "B2").upper()
                if area not in ("B2", "B4"):
                    area = "B2"
                b64 = str(body.get("image_base64") or "")
                if "," in b64 and b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                raw = base64.b64decode(b64)
                img_bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    self._send_json(400, {"ok": False, "error": "Could not decode image payload."})
                    return
                filename = str(body.get("filename") or f"custom_{area.lower()}.jpg")
                with _SERVER_LOCK:
                    res = analyze_lot(
                        area_code=area,
                        img_bgr=img_bgr,
                        out_dir=out_dir,
                        models_dir=models_dir,
                        prefix=f"custom_{area.lower()}_",
                        is_custom=True,
                    )
                    res.pop("map_img", None)
                    res["source_filename"] = filename
                    # Update CSV & combined report with latest uploaded lot result
                    other_area = "B4" if area == "B2" else "B2"
                    b2_slots = res["slots"] if area == "B2" else state_holder["report"]["lots"]["B2"]["slots"]
                    b4_slots = res["slots"] if area == "B4" else state_holder["report"]["lots"]["B4"]["slots"]
                    F4.write_occupancy_csv(res["slots"], out_dir / f"custom_{area.lower()}_occupancy.csv")
                    F4.write_occupancy_csv(b2_slots + b4_slots, out_dir / "lot_b2_b4_occupancy.csv")
                    res["ok"] = True
                self._send_json(200, res)
            except Exception as exc:
                logger.exception("Error handling /api/analyze")
                self._send_json(500, {"ok": False, "error": str(exc)})

    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info("Lot B2 & B4 4-model HTTP server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze Lot B2 & Lot B4 locally using 4-model fusion.")
    ap.add_argument("--b2-image", type=Path, default=ROOT_DIR / "ui/samples/area_b2_raw.jpg")
    ap.add_argument("--b4-image", type=Path, default=None, help="Optional raw image for Lot B4")
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "outputs/lots_b2_b4")
    ap.add_argument("--models", type=Path, default=ROOT_DIR / "models")
    ap.add_argument("--serve", action="store_true", help="Start HTTP server after running initial audit")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    report = run_both_lots(args.b2_image, args.b4_image, args.out, args.models, sync_ui_samples=True)
    print_summary(report)

    if args.serve:
        _serve_http(args.host, args.port, args.out, args.models, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
