#!/usr/bin/env python3
"""
scripts/run_area_b2_local.py
============================
End-to-end local runner for Area B2 (29 parking slots) combining:
  1. Pure-rotation pinhole homography & slot mapping (scripts/area_b2/topview_slots.py)
  2. Dual pretrained AI cross-check: YOLOX-L + MiDaS-small depth residual (scripts/area_b2/parking_ai_check.py)
  3. SENTRY occupancy pipeline classification, smoothing, freshness & mock event emission
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
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
from parking_occupancy.geometry import PhysicalStatus, PolygonMap  # noqa: E402
from parking_occupancy.inference import Detection, VehicleClass  # noqa: E402
from parking_occupancy.smoothing import StateSmoother  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("area_b2_local")


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


def _max_box_score_near_anchor(boxes: np.ndarray, anchor: tuple[int, int], pad: int = 12) -> float:
    """Return max YOLOX confidence among boxes containing anchor (±pad), or 0.0."""
    best = 0.0
    for b in boxes:
        if b[0] - pad <= anchor[0] <= b[2] + pad and b[1] - pad <= anchor[1] <= b[3] + pad:
            best = max(best, float(b[4]))
    return best


def run_area_b2_analysis(
    image_path: Path,
    config_path: Path,
    out_dir: Path,
    models_dir: Path,
    z_thresh: float = 4.0,
) -> dict[str, Any]:
    """Run topview warp, dual-model AI check, and SENTRY mock pipeline locally."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read Area B2 image: {image_path}")
    h, w = img.shape[:2]

    # --- 1. TopView geometry warp & visualizations ---
    tv = ts.TopView(w, h)
    top = ts.render_topview(img, tv)
    top_plain = top.copy()
    ts.scale_bar(top_plain, ts.PPM)
    cv2.imwrite(str(out_dir / "parking_topview.jpg"), top_plain, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_dir / "parking_slotmap_topview.jpg"), ts.slot_map_on_topview(top, tv), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_dir / "parking_slotmap_photo.jpg"), ts.slot_map_on_photo(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(out_dir / "parking_slotmap_plan.png"), ts.slot_plan())

    # --- 2. Dual pretrained AI models: YOLOX-L + MiDaS-small ---
    yolox_path = pac.get_model("yolox_l.onnx", str(models_dir))
    midas_path = pac.get_model("midas_small.onnx", str(models_dir))
    boxes = pac.detector_boxes(img, yolox_path)
    depth_model = pac.Depth(midas_path)
    near_res, s_near, o_near = pac.ground_plane_residual(depth_model, img, pac.NEAR_BAND, pac.GROUND_NEAR)
    far_res, s_far, o_far = pac.ground_plane_residual(depth_model, img, pac.FAR_CROP, pac.GROUND_FAR)

    # Evaluate per-slot dual-model evidence & build SENTRY detections
    rows: list[dict[str, Any]] = []
    detections: list[Detection] = []
    vis = img.copy()

    for sid, zone, plan_status, anchor, note in ts.SLOTS:
        if anchor is None:
            rows.append({
                "slot": sid, "zone": zone, "status": plan_status,
                "detector_box": 0, "height_z": None, "depth_pass": "",
                "agreement": "none", "note": "not visible in photo",
            })
            continue

        in_far = (
            pac.FAR_CROP[0] <= anchor[0] < pac.FAR_CROP[2]
            and pac.FAR_CROP[1] <= anchor[1] < pac.FAR_CROP[3]
            and anchor[1] < 760
        )
        z = pac.height_z(*((far_res, s_far, o_far) if in_far else (near_res, s_near, o_near)), anchor)
        det_score = _max_box_score_near_anchor(boxes, anchor)
        det_hit = det_score > 0.0
        up_hit = z >= z_thresh
        verdict = "both" if det_hit and up_hit else "one" if (det_hit or up_hit) else "none"

        rows.append({
            "slot": sid, "zone": zone, "status": plan_status,
            "detector_box": int(det_hit), "height_z": round(z, 1),
            "depth_pass": "far" if in_far else "near",
            "agreement": verdict, "note": note,
        })

        # Draw agreement badge (green=both, orange=one, red=none)
        col = {"both": (60, 170, 60), "one": (0, 170, 255), "none": (60, 60, 220)}[verdict]
        cv2.circle(vis, tuple(map(int, anchor)), 25, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(map(int, anchor)), 22, col, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(sid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(vis, sid, (int(anchor[0]) - tw // 2, int(anchor[1]) + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        # Positive-only depth trust: any slot backed by YOLOX OR MiDaS produces an anchor Detection
        if det_hit or up_hit:
            fused_conf = min(0.98, max(0.88, det_score)) if verdict == "both" else (0.64 if plan_status == "probable" else 0.78)
            detections.append(Detection(
                vehicle_class=VehicleClass.CAR,
                confidence=round(fused_conf, 3),
                x1=float(anchor[0] - 16),
                y1=float(anchor[1] - 16),
                x2=float(anchor[0] + 16),
                y2=float(anchor[1] + 16),
            ))

    # Write MiDaS heightmap, dual-model check image & CSV
    heat = np.zeros(img.shape[:2], np.float32)
    heat[pac.NEAR_BAND[1]:pac.NEAR_BAND[3]] = np.clip(near_res / (6.0 * s_near), 0.0, 1.0)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(out_dir / "parking_ai_heightmap.jpg"), cv2.addWeighted(img, 0.55, colour, 0.45, 0), [cv2.IMWRITE_JPEG_QUALITY, 92])
    mask = (heat > 0.02)[..., None]
    vis_out = np.where(mask, cv2.addWeighted(vis, 0.8, colour, 0.2, 0), vis)
    cv2.imwrite(str(out_dir / "parking_ai_check.jpg"), vis_out, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(out_dir / "parking_ai_check.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["slot", "zone", "status", "detector_box", "height_z", "depth_pass", "agreement", "note"])
        for r in rows:
            wr.writerow([
                r["slot"], r["zone"], r["status"],
                "" if r["height_z"] is None else r["detector_box"],
                "" if r["height_z"] is None else r["height_z"],
                r["depth_pass"], r["agreement"], r["note"],
            ])

    # --- 3. SENTRY pipeline classification + mock event emission ---
    capture_cfg = CaptureConfig.from_yaml(config_path)
    polygon_cfg_path = capture_cfg.polygon_config
    if not polygon_cfg_path.is_absolute():
        polygon_cfg_path = ROOT_DIR / polygon_cfg_path
    polygon_map = PolygonMap.from_json(polygon_cfg_path)
    meta_by_uuid = json.loads(polygon_cfg_path.read_text(encoding="utf-8"))["spaces"]
    slot_num_by_uuid = {s["space_id"]: s["slot_number"] for s in meta_by_uuid}

    smoother = StateSmoother.from_config(capture_cfg)
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
    # Force smoothing confirmation across n_avail frames for still-frame available bays if needed
    smoothed = smoother.update(frame.timestamp, raw_states)
    final_states = freshness.apply(frame.timestamp, smoothed)
    emitter.emit_changes(frame.timestamp, final_states)
    emitter.flush()
    rec_client.close()

    state_by_slot = {slot_num_by_uuid[s.space_id]: s for s in final_states}
    for r in rows:
        st = state_by_slot[r["slot"]]
        r["sentry_status"] = st.physical_status.value
        r["sentry_confidence"] = st.confidence

    tally: dict[str, int] = {}
    for r in rows:
        tally[r["agreement"]] = tally.get(r["agreement"], 0) + 1

    zone_summary = ts.summary()
    report = {
        "image": str(image_path),
        "area_m": [round(float(tv.size_m[0]), 1), round(float(tv.size_m[1]), 1)],
        "agreement_tally": tally,
        "weak_support_slots": [r["slot"] for r in rows if r["agreement"] != "both"],
        "zone_summary": zone_summary,
        "slots": rows,
        "emitted_events_count": len(rec_client.events),
        "sample_event": rec_client.events[0] if rec_client.events else None,
    }
    (out_dir / "sentry_area_b2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


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
          f"Right={zs['right']['seen']}/{zs['right']['slots']} (8 confirmed + 2 probable)  "
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run SENTRY Area B2 29-slot analysis completely locally.")
    ap.add_argument("--image", type=Path, default=ROOT_DIR / "ui/samples/area_b2_raw.jpg")
    ap.add_argument("--config", type=Path, default=ROOT_DIR / "configs/cameras/area_b2.yaml")
    ap.add_argument("--out", type=Path, default=ROOT_DIR / "outputs/area_b2")
    ap.add_argument("--models", type=Path, default=ROOT_DIR / "models")
    ap.add_argument("--z", type=float, default=4.0, help="MiDaS height z-score threshold")
    args = ap.parse_args()

    report = run_area_b2_analysis(args.image, args.config, args.out, args.models, z_thresh=args.z)
    print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
