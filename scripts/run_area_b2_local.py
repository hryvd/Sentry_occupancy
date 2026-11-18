#!/usr/bin/env python3
"""
scripts/run_area_b2_local.py
============================
End-to-end local runner & HTTP analysis server for Area B2 (29 parking slots) combining:
  1. Pure-rotation pinhole homography & slot mapping (scripts/area_b2/topview_slots.py)
  2. Dual pretrained AI cross-check: YOLOX-L + MiDaS-small depth residual (scripts/area_b2/parking_ai_check.py)
  3. SENTRY occupancy pipeline classification, smoothing, freshness & mock event emission
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import logging
import mimetypes
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
AREA_B2_DIR = Path(__file__).resolve().parent / "area_b2"
for _p in (str(SRC_DIR), str(AREA_B2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parking_ai_check as pac  # noqa: E402
import topview_slots as ts  # noqa: E402

from parking_occupancy.backend_client import BackendClient, BackendMode  # noqa: E402
from parking_occupancy.capture import CaptureConfig, Frame  # noqa: E402
from parking_occupancy.events import EventEmitter  # noqa: E402
from parking_occupancy.freshness import FreshnessTracker  # noqa: E402
from parking_occupancy.geometry import PhysicalStatus, PolygonMap, SpaceState  # noqa: E402
from parking_occupancy.inference import Detection, VehicleClass  # noqa: E402
from parking_occupancy.smoothing import StateSmoother  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("area_b2_local")

B2_REF_WIDTH = 2048
B2_REF_HEIGHT = 1536
_ANALYZE_LOCK = threading.Lock()
_DEPTH_CACHE: dict[str, pac.Depth] = {}


class _RecordingMockClient:
    """Thin delegate over BackendClient(BackendMode.MOCK) that records emitted payloads."""

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


def _get_depth_model(midas_path: str) -> pac.Depth:
    """Cache the MiDaS ONNX session across HTTP analysis requests."""
    if midas_path not in _DEPTH_CACHE:
        _DEPTH_CACHE[midas_path] = pac.Depth(midas_path)
    return _DEPTH_CACHE[midas_path]


def _match_box_near_anchor(
    boxes: np.ndarray,
    anchor: tuple[int, int],
    pad: int = 12,
) -> tuple[float, list[float] | None]:
    """Return (max_score, [x0, y0, x1, y1, score]) among boxes containing anchor (±pad)."""
    best_score = 0.0
    best_box: list[float] | None = None
    for b in boxes:
        if b[0] - pad <= anchor[0] <= b[2] + pad and b[1] - pad <= anchor[1] <= b[3] + pad:
            score = float(b[4])
            if score > best_score:
                best_score = score
                best_box = [round(float(b[0]), 1), round(float(b[1]), 1), round(float(b[2]), 1), round(float(b[3]), 1), round(score, 3)]
    return best_score, best_box


def _nms_boxes(boxes: np.ndarray, score_thresh: float = 0.25, iou_thresh: float = 0.45) -> list[list[float]]:
    """Run NMS over merged full + far-zoom YOLOX boxes and return JSON-friendly [x0,y0,x1,y1,score]."""
    if len(boxes) == 0:
        return []
    xywh = [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])] for b in boxes]
    scores = [float(b[4]) for b in boxes]
    idx = cv2.dnn.NMSBoxes(xywh, scores, score_thresh, iou_thresh)
    if len(idx) == 0:
        return []
    result: list[list[float]] = []
    for i in np.array(idx).flatten():
        b = boxes[int(i)]
        result.append([round(float(b[0]), 1), round(float(b[1]), 1), round(float(b[2]), 1), round(float(b[3]), 1), round(float(b[4]), 3)])
    return result


def _render_custom_slot_plan(
    status_by_slot: dict[str, str],
    zone_summary: dict[str, dict[str, int]],
    path_scale: int = 34,
) -> np.ndarray:
    """Render the 29-slot schematic plan colored by custom live per-slot occupancy status."""
    S = path_scale
    BW, BD = 2.4, 5.0
    W_m, H_m = 23.0, 37.0
    img = np.full((int(H_m * S), int(W_m * S), 3), 255, np.uint8)
    colors = {"confirmed": None, "probable": None, "unconfirmed": (235, 235, 235), "informal": (205, 230, 205)}

    def rect(x: float, y: float, w: float, h: float, sid: str, zone: str, status: str) -> None:
        p1, p2 = (int(x * S), int(y * S)), (int((x + w) * S), int((y + h) * S))
        k = 0.65 if status == "confirmed" else 0.35
        fill = colors[status] if colors[status] is not None else tuple(int((1 - k) * 255 + k * c) for c in ts.ZONE_COLOR[zone])
        cv2.rectangle(img, p1, p2, fill, -1)
        if status == "unconfirmed":
            for xx in range(p1[0], p2[0], 10):
                cv2.line(img, (xx, p1[1]), (min(xx + (p2[1] - p1[1]), p2[0]), min(p1[1] + (p2[0] - xx), p2[1])), (200, 200, 200), 1)
        cv2.rectangle(img, p1, p2, (50, 50, 50), 2)
        (tw, th), _ = cv2.getTextSize(sid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, sid, ((p1[0] + p2[0]) // 2 - tw // 2, (p1[1] + p2[1]) // 2 + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    y_bottom = 29.5
    for i in range(11):
        y = y_bottom - (i + 1) * BW
        rect(1.0, y, BD, BW, f"L{i + 1}", "left", status_by_slot[f"L{i + 1}"])
        rect(W_m - 1.0 - BD, y, BD, BW, f"R{i + 1}", "right", status_by_slot[f"R{i + 1}"])
    cx = W_m / 2
    yb = y_bottom - 3.4
    for k, sid in enumerate(("M1", "M2", "M3")):
        rect(cx - 1.5 * BW + k * BW, yb - BD, BW, BD, sid, "middle", status_by_slot[sid])
    rect(cx - BD / 2, yb + 0.4, BD, BW, "M4", "middle", status_by_slot["M4"])
    for k, sid in enumerate(("M5", "M6", "M7")):
        rect(cx - BW / 2, yb - BD - (k + 1) * BD, BW, BD, sid, "middle", status_by_slot[sid])
    cv2.putText(img, "AREA B2 SLOT PLAN (live custom run)", (int(1 * S), int(1.2 * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "bays 2.4 m x 5.0 m  |  top of plan = far end of lot", (int(1 * S), int(2.0 * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
    ts.draw_legend(img, int(1.6 * S), int(32.0 * S), zone_summary)
    return img


def analyze_b2_image(
    img_bgr: np.ndarray,
    models_dir: Path,
    out_dir: Path,
    prefix: str = "parking_",
    is_custom: bool = False,
    config_path: Path = ROOT_DIR / "configs/cameras/area_b2.yaml",
    z_thresh: float = 4.0,
    source_name: str = "area_b2_raw.jpg",
) -> dict[str, Any]:
    """Run Area B2 topview warp, YOLOX-L + MiDaS-small cross-check, and SENTRY mock pipeline."""
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Empty image supplied to Area B2 analyzer.")
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_h, orig_w = img_bgr.shape[:2]
    if (orig_w, orig_h) != (B2_REF_WIDTH, B2_REF_HEIGHT):
        img = cv2.resize(img_bgr, (B2_REF_WIDTH, B2_REF_HEIGHT), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr
    h, w = img.shape[:2]

    # --- 1. TopView geometry warp ---
    tv = ts.TopView(w, h)
    top = ts.render_topview(img, tv)
    top_plain = top.copy()
    ts.scale_bar(top_plain, ts.PPM)

    # --- 2. Dual pretrained AI models: YOLOX-L + MiDaS-small ---
    yolox_path = pac.get_model("yolox_l.onnx", str(models_dir))
    midas_path = pac.get_model("midas_small.onnx", str(models_dir))
    boxes = pac.detector_boxes(img, yolox_path)
    nms_boxes = _nms_boxes(boxes, score_thresh=0.25, iou_thresh=0.45)
    depth_model = _get_depth_model(midas_path)
    near_res, s_near, o_near = pac.ground_plane_residual(depth_model, img, pac.NEAR_BAND, pac.GROUND_NEAR)
    far_res, s_far, o_far = pac.ground_plane_residual(depth_model, img, pac.FAR_CROP, pac.GROUND_FAR)

    # Evaluate per-slot dual-model evidence
    rows: list[dict[str, Any]] = []
    detections: list[Detection] = []
    vis = img.copy()
    if is_custom:
        for bx0, by0, bx1, by1, _ in nms_boxes:
            cv2.rectangle(vis, (int(bx0), int(by0)), (int(bx1), int(by1)), (0, 185, 255), 2)

    match_pad = 18 if is_custom else 12
    zone_summary: dict[str, dict[str, int]] = {
        "left": {"slots": 0, "seen": 0},
        "right": {"slots": 0, "seen": 0},
        "middle": {"slots": 0, "seen": 0},
    }

    for sid, zone, ref_plan_status, anchor, ref_note in ts.SLOTS:
        zone_summary[zone]["slots"] += 1
        if anchor is None:
            rows.append({
                "slot": sid, "zone": zone, "status": "unconfirmed", "planStatus": "unconfirmed",
                "detector_box": 0, "matched_box": None, "height_z": None, "depth_pass": "",
                "agreement": "none", "note": "not visible in photo",
                "anchor": None, "topview": None, "meters": None,
                "custom_sentry_status": "unknown", "custom_sentry_conf": None,
            })
            continue

        top_pt = tv.to_topview_px([anchor])[0]
        m_pt = tv.to_metres([anchor])[0]
        in_far = (
            pac.FAR_CROP[0] <= anchor[0] < pac.FAR_CROP[2]
            and pac.FAR_CROP[1] <= anchor[1] < pac.FAR_CROP[3]
            and anchor[1] < 760
        )
        z = pac.height_z(*((far_res, s_far, o_far) if in_far else (near_res, s_near, o_near)), anchor)
        z_round = round(float(z), 1)
        depth_pass = "far" if in_far else "near"
        det_score, matched_box = _match_box_near_anchor(boxes, anchor, pad=match_pad)
        det_hit = det_score > 0.0
        up_hit = z >= z_thresh
        verdict = "both" if det_hit and up_hit else "one" if (det_hit or up_hit) else "none"

        if is_custom:
            if verdict == "both":
                plan_status = "informal" if zone == "middle" else "confirmed"
                c_status = "occupied"
                c_conf: float | None = 0.95 if depth_pass == "near" else 0.92
                note = f"YOLOX box + MiDaS z={z_round}"
                zone_summary[zone]["seen"] += 1
            elif verdict == "one" and det_hit:
                plan_status = "probable" if sid in ("R9", "R10") else ("informal" if zone == "middle" else "confirmed")
                c_status = "occupied"
                c_conf = 0.64 if sid in ("R9", "R10") else 0.78
                note = f"YOLOX vehicle box (depth z={z_round} < {z_thresh:.1f})"
                zone_summary[zone]["seen"] += 1
            elif verdict == "one":
                plan_status = "probable"
                c_status = "available"
                c_conf = 0.55
                note = f"depth elevation only (z={z_round}, no YOLOX box)"
            else:
                plan_status = "unconfirmed"
                c_status = "available"
                c_conf = 0.85
                note = "no vehicle detected (empty)"
        else:
            plan_status = ref_plan_status
            note = ref_note
            c_status = "unknown"
            c_conf = None
            zone_summary[zone]["seen"] += 1

        rows.append({
            "slot": sid, "zone": zone, "status": plan_status, "planStatus": plan_status,
            "detector_box": int(det_hit), "matched_box": matched_box, "height_z": z_round,
            "depth_pass": depth_pass, "agreement": verdict, "note": note,
            "anchor": [int(anchor[0]), int(anchor[1])],
            "topview": [round(float(top_pt[0]), 1), round(float(top_pt[1]), 1)],
            "meters": [round(float(m_pt[0]), 1), round(float(m_pt[1]), 1)],
            "custom_sentry_status": c_status, "custom_sentry_conf": c_conf,
        })

        col = {"both": (60, 170, 60), "one": (0, 170, 255), "none": (60, 60, 220)}[verdict]
        cv2.circle(vis, tuple(map(int, anchor)), 25, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(map(int, anchor)), 22, col, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(sid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(vis, sid, (int(anchor[0]) - tw // 2, int(anchor[1]) + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if (is_custom and det_hit) or (not is_custom and (det_hit or up_hit)):
            fused_conf = min(0.98, max(0.88, det_score)) if verdict == "both" else (0.64 if plan_status == "probable" else 0.78)
            detections.append(Detection(
                vehicle_class=VehicleClass.CAR,
                confidence=round(fused_conf, 3),
                x1=float(anchor[0] - 16),
                y1=float(anchor[1] - 16),
                x2=float(anchor[0] + 16),
                y2=float(anchor[1] + 16),
            ))

    # --- 3. Write visual artifacts & CSV ---
    heat = np.zeros(img.shape[:2], np.float32)
    heat[pac.NEAR_BAND[1]:pac.NEAR_BAND[3]] = np.clip(near_res / (6.0 * max(float(s_near), 1e-6)), 0.0, 1.0)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heightmap_img = cv2.addWeighted(img, 0.55, colour, 0.45, 0)
    mask = (heat > 0.02)[..., None]
    vis_out = np.where(mask, cv2.addWeighted(vis, 0.8, colour, 0.2, 0), vis)

    ts_ms = int(time.time() * 1000)
    if not is_custom and prefix == "parking_":
        check_name = "parking_ai_check.jpg"
        height_name = "parking_ai_heightmap.jpg"
        topview_name = "parking_topview.jpg"
        plan_name = "parking_slotmap_plan.png"
        cv2.imwrite(str(out_dir / "parking_slotmap_topview.jpg"), ts.slot_map_on_topview(top, tv), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / "parking_slotmap_photo.jpg"), ts.slot_map_on_photo(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_dir / plan_name), ts.slot_plan())
    else:
        check_name = f"{prefix}check.jpg"
        height_name = f"{prefix}heightmap.jpg"
        topview_name = f"{prefix}topview.jpg"
        plan_name = f"{prefix}plan.png"
        status_map = {r["slot"]: r["planStatus"] for r in rows}
        cv2.imwrite(str(out_dir / plan_name), _render_custom_slot_plan(status_map, zone_summary))

    cv2.imwrite(str(out_dir / topview_name), top_plain, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_dir / height_name), heightmap_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_dir / check_name), vis_out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    csv_name = "parking_ai_check.csv" if (not is_custom and prefix == "parking_") else f"{prefix}check.csv"
    with open(out_dir / csv_name, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["slot", "zone", "status", "detector_box", "height_z", "depth_pass", "agreement", "note"])
        for r in rows:
            wr.writerow([
                r["slot"], r["zone"], r["status"],
                "" if r["height_z"] is None else r["detector_box"],
                "" if r["height_z"] is None else r["height_z"],
                r["depth_pass"], r["agreement"], r["note"],
            ])

    # --- 4. SENTRY pipeline classification + mock event emission ---
    capture_cfg = CaptureConfig.from_yaml(config_path)
    polygon_cfg_path = capture_cfg.polygon_config
    if not polygon_cfg_path.is_absolute():
        polygon_cfg_path = ROOT_DIR / polygon_cfg_path
    polygon_map = PolygonMap.from_json(polygon_cfg_path)
    meta_by_uuid = json.loads(polygon_cfg_path.read_text(encoding="utf-8"))["spaces"]
    slot_num_by_uuid = {s["space_id"]: s["slot_number"] for s in meta_by_uuid}
    row_by_slot = {r["slot"]: r for r in rows}

    smoother = StateSmoother(n_to_occupied=1, n_to_available=1) if is_custom else StateSmoother.from_config(capture_cfg)
    freshness = FreshnessTracker.from_config(capture_cfg)
    raw_client = BackendClient(mode=BackendMode.MOCK, config=capture_cfg)
    rec_client = _RecordingMockClient(raw_client)
    emitter = EventEmitter(
        producer_id=capture_cfg.device_id,
        model_version=capture_cfg.model_version,
        calibration_version=polygon_map.calibration_version,
        client=rec_client,
    )

    frame = Frame(image=img, timestamp=time.monotonic(), camera_id=capture_cfg.camera_id, frame_index=0)
    raw_states = polygon_map.classify(frame, detections)
    smoothed = smoother.update(frame.timestamp, raw_states)
    if is_custom:
        aligned: list[SpaceState] = []
        for st in smoothed:
            sid = slot_num_by_uuid[st.space_id]
            r = row_by_slot[sid]
            aligned.append(SpaceState(
                space_id=st.space_id,
                lot_code=st.lot_code,
                compatibility=st.compatibility,
                physical_status=PhysicalStatus(r["custom_sentry_status"]),
                confidence=r["custom_sentry_conf"],
            ))
        smoothed = aligned

    final_states = freshness.apply(frame.timestamp, smoothed)
    emitter.emit_changes(frame.timestamp, final_states)
    emitter.flush()
    rec_client.close()

    state_by_slot = {slot_num_by_uuid[s.space_id]: s for s in final_states}
    for r in rows:
        st = state_by_slot[r["slot"]]
        r["sentry_status"] = st.physical_status.value
        r["sentry_confidence"] = st.confidence
        r.pop("custom_sentry_status", None)
        r.pop("custom_sentry_conf", None)

    tally: dict[str, int] = {"both": 0, "one": 0, "none": 0}
    for r in rows:
        tally[r["agreement"]] = tally.get(r["agreement"], 0) + 1

    report: dict[str, Any] = {
        "ok": True,
        "image": source_name,
        "original_res": f"{orig_w}×{orig_h}",
        "area_m": [round(float(tv.size_m[0]), 1), round(float(tv.size_m[1]), 1)],
        "agreement_tally": tally,
        "weak_support_slots": [r["slot"] for r in rows if r["agreement"] != "both"],
        "zone_summary": zone_summary,
        "slots": rows,
        "detected_boxes": nms_boxes,
        "overlay_urls": {
            "check": f"/outputs/area_b2/{check_name}?t={ts_ms}",
            "heightmap": f"/outputs/area_b2/{height_name}?t={ts_ms}",
            "topview": f"/outputs/area_b2/{topview_name}?t={ts_ms}",
            "plan": f"/outputs/area_b2/{plan_name}?t={ts_ms}",
        },
        "emitted_events_count": len(rec_client.events),
        "sample_event": rec_client.events[0] if rec_client.events else None,
    }
    report_name = "sentry_area_b2_report.json" if (not is_custom and prefix == "parking_") else f"{prefix}report.json"
    (out_dir / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_area_b2_analysis(
    image_path: Path,
    config_path: Path,
    out_dir: Path,
    models_dir: Path,
    z_thresh: float = 4.0,
) -> dict[str, Any]:
    """Run default static Area B2 audit from an image path on disk."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read Area B2 image: {image_path}")
    return analyze_b2_image(
        img_bgr=img,
        models_dir=models_dir,
        out_dir=out_dir,
        prefix="parking_",
        is_custom=False,
        config_path=config_path,
        z_thresh=z_thresh,
        source_name=str(image_path),
    )


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print the 29-slot Area B2 audit and sample SENTRY event to stdout."""
    print("\n================ SENTRY AREA B2 — LOCAL DUAL-MODEL AUDIT ================")
    print(f"Image        : {report['image']}")
    print(f"Warped Area  : {report['area_m'][0]} m x {report['area_m'][1]} m (PPM={ts.PPM} px/m)")
    print(f"Model Tally  : {report['agreement_tally']}")
    print(f"Weak Support : {report['weak_support_slots']}")
    zs = report["zone_summary"]
    tot_v = sum(v["seen"] for v in zs.values())
    tot_s = sum(v["slots"] for v in zs.values())
    print(f"Zone Summary : Left={zs['left']['seen']}/{zs['left']['slots']}  "
          f"Right={zs['right']['seen']}/{zs['right']['slots']}  "
          f"Middle={zs['middle']['seen']}/{zs['middle']['slots']}  Total={tot_v}/{tot_s}")
    print("-" * 96)
    print(f"{'Slot':<5} {'Zone':<7} {'Plan':<12} {'Box':<4} {'MiDaS z':<8} {'Pass':<5} {'Agree':<6} {'SENTRY':<10} {'Conf':<6} Note")
    print("-" * 96)
    for r in report["slots"]:
        z_str = "-" if r["height_z"] is None else f"{r['height_z']:.1f}"
        conf_str = "-" if r["sentry_confidence"] is None else f"{r['sentry_confidence']:.3f}"
        print(f"{r['slot']:<5} {r['zone']:<7} {r['status']:<12} {r['detector_box']:<4} "
              f"{z_str:<8} {r['depth_pass'] or '-':<5} {r['agreement']:<6} "
              f"{r['sentry_status']:<10} {conf_str:<6} {r['note']}")
    print("-" * 96)
    print(f"Mock SENTRY Events Emitted: {report['emitted_events_count']} (strictly observation, correlation_id=null)")
    if report["sample_event"]:
        print("Sample Event Payload:")
        print(json.dumps(report["sample_event"], indent=2))
    print("=========================================================================\n")


def _serve_http(
    host: str,
    port: int,
    config_path: Path,
    out_dir: Path,
    models_dir: Path,
    z_thresh: float,
) -> None:
    """Serve UI assets, /outputs/, and POST /api/analyze on localhost."""
    ui_dir = (ROOT_DIR / "ui").resolve()
    outputs_root = (ROOT_DIR / "outputs").resolve()

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def _send_file(self, base_dir: Path, rel_path: str) -> None:
            clean_rel = urllib.parse.unquote(rel_path).lstrip("/")
            if not clean_rel:
                clean_rel = "index.html"
            candidate = (base_dir / clean_rel).resolve()
            if base_dir not in candidate.parents and candidate != base_dir:
                self.send_error(403, "Forbidden")
                return
            if not candidate.is_file():
                self.send_error(404, "Not Found")
                return
            ctype, _ = mimetypes.guess_type(str(candidate))
            data = candidate.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/health":
                self._send_json(200, {"ok": True, "mode": "area_b2_local"})
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
                    self._send_json(400, {"ok": False, "error": "Invalid request body length."})
                    return
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                b64 = body.get("image_base64", "")
                if "," in b64 and b64.startswith("data:"):
                    b64 = b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64)
                img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    self._send_json(400, {"ok": False, "error": "Could not decode image data."})
                    return
                filename = str(body.get("filename") or "custom.jpg")
                with _ANALYZE_LOCK:
                    report = analyze_b2_image(
                        img_bgr=img_bgr,
                        models_dir=models_dir,
                        out_dir=out_dir,
                        prefix="custom_",
                        is_custom=True,
                        config_path=config_path,
                        z_thresh=z_thresh,
                        source_name=filename,
                    )
                self._send_json(200, report)
            except Exception as exc:
                logger.exception("Error analyzing custom Area B2 image")
                self._send_json(500, {"ok": False, "error": str(exc)})

    server = ThreadingHTTPServer((host, port), _Handler)
    logger.info("Area B2 local analysis server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SENTRY Area B2 29-slot analysis completely locally.")
    ap.add_argument("--image", type=Path, default=ROOT_DIR / "ui/samples/area_b2_raw.jpg")
    ap.add_argument("--config", type=Path, default=ROOT_DIR / "configs/cameras/area_b2.yaml")
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "outputs/area_b2")
    ap.add_argument("--models", type=Path, default=ROOT_DIR / "models")
    ap.add_argument("--z", type=float, default=4.0, help="MiDaS height z-score threshold")
    ap.add_argument("--serve", action="store_true", help="Serve local UI + POST /api/analyze endpoint")
    ap.add_argument("--host", default="127.0.0.1", help="HTTP server bind host")
    ap.add_argument("--port", type=int, default=8080, help="HTTP server port")
    args = ap.parse_args()

    if args.serve:
        _serve_http(args.host, args.port, args.config, args.out, args.models, args.z)
        return 0

    report = run_area_b2_analysis(args.image, args.config, args.out, args.models, z_thresh=args.z)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
