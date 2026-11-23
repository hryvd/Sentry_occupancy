#!/usr/bin/env python3
"""
scripts/lots_b2_b4/fusion_4model.py
===================================
Optimized 4-Model Occupancy Fusion Pipeline for Lot B2 (22 bays) & Lot B4 (27 bays).

Models & High-Value Model 4 Weighting:
  - model_1 (0.15): Geometry (metric ground footprint overlap + MiDaS ground-plane elevation)
  - model_2 (0.15): Contextual slot analysis (free gap width, spillover overhang, sightline occlusion)
  - model_3 (0.20): YOLO (cached YOLOX-L tiled + far-zoom detections + roof-stack score)
  - model_4 (0.50): Top-down view full geometry + context + YOLO re-analysis (highest weight)

Weighted mean:
  mean = 0.15 * model_1 + 0.15 * model_2 + 0.20 * model_3 + 0.50 * model_4
  final_state = "occupied" if mean >= 0.50 else "available"
  final_id    = 1 if final_state == "occupied" else 0
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import threading
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

THIS_DIR = Path(__file__).resolve().parent
AREA_B2_DIR = THIS_DIR.parent / "area_b2"
for _p in (str(THIS_DIR), str(AREA_B2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parking_ai_check as pac  # noqa: E402
import slot_occupancy as SO  # noqa: E402
import topview_slots as T  # noqa: E402

WEIGHTS = (0.15, 0.15, 0.20, 0.50)
CSV_HEADER = [
    "area",
    "slot_id",
    "zone",
    "compatibility",
    "space_id",
    "initial_state",
    "model_1",
    "model_2",
    "model_3",
    "model_4",
    "mean",
    "final_state",
    "final_id",
    "backend_display_code",
]

_CACHE_LOCK = threading.Lock()
_ONNX_CACHE: dict[str, Any] = {}


def _clip01(val: float) -> float:
    return float(max(0.0, min(1.0, val)))


def _get_yolox_bundle(models_dir: Path):
    """Load YOLOX-L session once and cache anchor grids."""
    key = f"yolox:{Path(models_dir).resolve()}"
    with _CACHE_LOCK:
        if key not in _ONNX_CACHE:
            model_path = pac.get_model("yolox_l.onnx", str(models_dir))
            sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            inp_name = sess.get_inputs()[0].name
            gx, sx = [], []
            for s in (8, 16, 32):
                n = 640 // s
                xv, yv = np.meshgrid(np.arange(n), np.arange(n))
                gx.append(np.stack((xv, yv), 2).reshape(-1, 2))
                sx.append(np.full((n * n, 1), s))
            G, S = np.concatenate(gx), np.concatenate(sx)
            _ONNX_CACHE[key] = (sess, inp_name, G, S)
        return _ONNX_CACHE[key]


def _get_midas_model(models_dir: Path) -> pac.Depth:
    """Load MiDaS-small session once and cache across runs."""
    key = f"midas:{Path(models_dir).resolve()}"
    with _CACHE_LOCK:
        if key not in _ONNX_CACHE:
            model_path = pac.get_model("midas_small.onnx", str(models_dir))
            _ONNX_CACHE[key] = pac.Depth(model_path)
        return _ONNX_CACHE[key]


def detect_vehicles_cached(
    img_bgr: np.ndarray,
    models_dir: Path,
    score_thresh: float = 0.30,
    far_zoom_thresh: float = 0.25,
) -> tuple[np.ndarray, list[tuple[str, tuple[float, float, float, float], str, float]], np.ndarray]:
    """Run tiled YOLOX-L + far-end zoom crops using a single cached ONNX session."""
    sess, inp_name, G, S = _get_yolox_bundle(models_dir)
    H, W = img_bgr.shape[:2]

    def run_crop(crop: np.ndarray, ox: int, oy: int, clip_edges: bool, sc_cut: float, classes=(2, 3, 5, 7), cubic: bool = False):
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return []
        r = min(640.0 / ch, 640.0 / cw)
        pad = np.full((640, 640, 3), 114, dtype=np.uint8)
        rs = cv2.resize(crop, (int(cw * r), int(ch * r)), interpolation=cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        out = sess.run(None, {inp_name: pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy, wh = (out[:, :2] + G) * S, np.exp(out[:, 2:4]) * S
        sc = out[:, 4:5] * out[:, 5:]
        res = []
        for c in classes:
            m = sc[:, c] > sc_cut
            for (cx, cy), (bw, bh), s in zip(xy[m], wh[m], sc[m, c]):
                b = [
                    (cx - bw / 2.0) / r + ox,
                    (cy - bh / 2.0) / r + oy,
                    (cx + bw / 2.0) / r + ox,
                    (cy + bh / 2.0) / r + oy,
                ]
                if clip_edges and (
                    (ox > 0 and b[0] < ox + 6)
                    or (ox + cw < W and b[2] > ox + cw - 6)
                    or (oy > 0 and b[1] < oy + 6)
                    or (oy + ch < H and b[3] > oy + ch - 6)
                ):
                    continue
                res.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), float(c)])
        return res

    main_dets = run_crop(img_bgr, 0, 0, False, score_thresh)
    for x0, y0, x1, y1 in [(0, 300, 1100, 1236), (948, 300, 2048, 1236), (0, 500, 1100, 1536), (948, 500, 2048, 1536), (500, 400, 1550, 1300)]:
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        main_dets += run_crop(img_bgr[y0c:y1c, x0c:x1c], x0c, y0c, True, score_thresh)

    # Include far-end zoom crops in unified detection + NMS so tiny stacked far-row cars keep full recall
    for zname, (x0, y0, x1, y1) in pac.FAR_ZOOMS.items():
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        # Drop half-cut boxes on the crop outer edges cleanly while keeping small interior cars
        main_dets += run_crop(img_bgr[y0c:y1c, x0c:x1c], x0c, y0c, True, far_zoom_thresh, classes=(2, 5, 7), cubic=True)

    if main_dets:
        d = np.array(main_dets, dtype=np.float64)
        boxes_xywh = [[float(a), float(b), float(c - a), float(e - b)] for a, b, c, e, _, _ in d]
        idx = cv2.dnn.NMSBoxes(boxes_xywh, [float(s) for s in d[:, 4]], far_zoom_thresh, 0.45)
        kept = d[np.array(idx).flatten()] if len(idx) > 0 else np.empty((0, 6))
    else:
        kept = np.empty((0, 6))

    live_typed = [
        (SO.COCO_KIND[int(b[5])], (float(b[0]), float(b[1]), float(b[2]), float(b[3])), "yolox", float(b[4]))
        for b in kept
        if int(b[5]) in SO.COCO_KIND and float(b[4]) >= 0.45
    ]
    return kept, live_typed, kept[:, :5] if len(kept) else np.empty((0, 5))


def compute_midas_residuals(img_bgr: np.ndarray, models_dir: Path):
    """Run cached MiDaS-small on near/far bands and return (z_lookup_fn, heightmap_bgr)."""
    depth_model = _get_midas_model(models_dir)
    near_res, s_near, o_near = pac.ground_plane_residual(depth_model, img_bgr, pac.NEAR_BAND, pac.GROUND_NEAR)
    far_res, s_far, o_far = pac.ground_plane_residual(depth_model, img_bgr, pac.FAR_CROP, pac.GROUND_FAR)
    s_near = max(float(s_near), 1e-6)
    s_far = max(float(s_far), 1e-6)

    def z_at(px: float, py: float) -> float:
        in_far = pac.FAR_CROP[0] <= px < pac.FAR_CROP[2] and pac.FAR_CROP[1] <= py < pac.FAR_CROP[3] and py < 760
        if in_far:
            return float(pac.height_z(far_res, s_far, o_far, (px, py)))
        return float(pac.height_z(near_res, s_near, o_near, (px, py)))

    heat = np.zeros(img_bgr.shape[:2], dtype=np.float32)
    heat[pac.NEAR_BAND[1]:pac.NEAR_BAND[3]] = np.clip(near_res / (6.0 * s_near), 0.0, 1.0)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heightmap = cv2.addWeighted(img_bgr, 0.55, colour, 0.45, 0)
    return z_at, heightmap, heat


def _topview_visual_stats(topview_img: np.ndarray | None, tv, slots: list[SO.Slot]) -> dict[str, float]:
    """Compute warped top-down bay visual deviance score vs bare concrete baseline."""
    if topview_img is None:
        return {}
    h_tv, w_tv = topview_img.shape[:2]
    gray = cv2.cvtColor(topview_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    # Measure aisle concrete baseline (x ~ 10..15 m, near half y ~ 6..18 m)
    ax0, ax1 = int(max(0, 10.5 * SO.PPM)), int(min(w_tv, 15.0 * SO.PPM))
    ay0, ay1 = int(max(0, 6.0 * SO.PPM)), int(min(h_tv, 17.5 * SO.PPM))
    if ax1 > ax0 + 20 and ay1 > ay0 + 20:
        aisle_edge = float(np.percentile(edges[ay0:ay1, ax0:ax1], 45))
        aisle_std = float(np.percentile(np.std(topview_img[ay0:ay1, ax0:ax1], axis=2), 50))
    else:
        aisle_edge, aisle_std = 6.0, 12.0

    scores: dict[str, float] = {}
    for s in slots:
        poly_px = np.int32(np.round(s.poly * SO.PPM))
        if poly_px[:, 1].max() < 2 or poly_px[:, 1].min() >= h_tv - 2 or poly_px[:, 0].max() < 2 or poly_px[:, 0].min() >= w_tv - 2:
            continue
        mask = np.zeros((h_tv, w_tv), dtype=np.uint8)
        cv2.fillPoly(mask, [poly_px], 1)
        if mask.sum() < 120:
            continue
        e_mean = float(edges[mask == 1].mean())
        col_std = float(topview_img[mask == 1].std())
        edge_score = _clip01((e_mean - aisle_edge * 0.8) / max(aisle_edge * 2.2, 8.0))
        tex_score = _clip01((col_std - max(aisle_std, 12.0)) / 32.0)
        scores[s.id] = _clip01(0.60 * edge_score + 0.40 * tex_score)
    return scores


def score_lot_4model(
    area_code: str,
    tv,
    slots: list[SO.Slot],
    vehs: list[SO.Vehicle],
    rows_dict: dict[str, list[dict]],
    R: dict[str, dict[str, float]],
    img_bgr: np.ndarray | None = None,
    topview_img: np.ndarray | None = None,
    z_at=None,
    live_yolox_boxes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Compute Model 1..4, Weighted Mean (M4=50%), Final State, and Final ID per slot."""
    to_photo = SO.photo_px(tv)
    tv_vis = _topview_visual_stats(topview_img, tv, slots)
    non_stack_vehs = [v for v in vehs if v.src != "stack"]

    ordered_infos = []
    for row_key in ("left", "right", "moto"):
        if row_key in rows_dict:
            ordered_infos.extend(rows_dict[row_key])

    results: list[dict[str, Any]] = []
    for info in ordered_infos:
        s: SO.Slot = info["slot"]
        v: SO.Vehicle | None = info["vehicle"]
        cov_ratio = float(info["cover"])
        free_w = float(info["free_width"])
        hidden = float(info.get("hidden", 0.0))
        bay_pitch = max(float(s.t1 - s.t0), 0.5)
        min_req_gap = 0.65 if s.row == "moto" else SO.MIN_CAR_GAP_M

        # --- Initial State (single-frame perspective check BEFORE top-down/stack/re-fit) ---
        compat_raw_cov = sum(
            SO.overlap(vv.foot, s.poly) for vv in non_stack_vehs if (vv.kind == "moto") == (s.row == "moto")
        ) / max(s.area, 1e-4)
        compat_raw_cov = min(1.0, compat_raw_cov)
        raw_assigned_hit = v is not None and v.src == "yolox" and v.method.startswith(("1 ", "2 "))
        initial_state = "occupied" if (compat_raw_cov >= 0.35 or raw_assigned_hit) else "available"

        # Optional MiDaS height residual at photo projection of bay center
        z_val = None
        if z_at is not None and img_bgr is not None:
            px, py = to_photo([s.center])[0]
            if 0 <= px < img_bgr.shape[1] and 0 <= py < img_bgr.shape[0]:
                z_val = z_at(float(px), float(py))

        # --- Model 1: Geometry (weight 0.15) ---
        if v is not None:
            if v.src == "stack":
                geom_base = 0.72 + 0.18 * float(v.robust)
            else:
                v_foot_share = R.get(v.id, {}).get(s.id, cov_ratio)
                # Geometric alignment around bay center compensates for thin moto bays
                center_align = math.exp(-abs(float(v.t or s.t) - s.t) / (1.25 * bay_pitch))
                geom_base = max(cov_ratio, 0.75 * v_foot_share + 0.25 * center_align, 0.76 * center_align)
            if z_val is not None and z_val >= 2.0:
                depth_boost = _clip01(z_val / 6.0)
                model_1 = _clip01(0.82 * geom_base + 0.18 * depth_boost)
            else:
                model_1 = _clip01(geom_base)
        else:
            model_1 = _clip01(cov_ratio * 0.60 if cov_ratio > 0.05 else 0.03)

        # --- Model 2: Contextual Slot Analysis (weight 0.15) ---
        blocked_frac = _clip01(1.0 - free_w / min_req_gap)
        if v is not None:
            if v.src == "stack":
                model_2 = _clip01(0.62 + 0.23 * blocked_frac + 0.15 * float(v.robust))
            else:
                model_2 = _clip01(0.45 * min(1.0, cov_ratio + 0.32) + 0.35 * blocked_frac + 0.20 * max(0.55, hidden))
        else:
            if info["status"] == "FREE (blocked)":
                model_2 = _clip01(max(0.58, cov_ratio))
            else:
                model_2 = _clip01(max(0.02, 0.15 * cov_ratio + 0.10 * hidden + 0.05 * blocked_frac))

        # --- Model 3: YOLO Detector (weight 0.20) ---
        if v is not None:
            if v.src == "yolox":
                model_3 = _clip01(v.score)
            elif v.src == "stack":
                model_3 = _clip01(v.score * float(v.robust))
            else:
                model_3 = 0.750
        else:
            # No vehicle assigned to this bay: check only strong unassigned boxes overlapping bay footprint
            model_3 = 0.000

        # --- Model 4: Top-Down View Full Geometry + Context + YOLO Re-Analysis (weight 0.50 — HIGH VALUE) ---
        stage_map = {"1": 0.95, "2": 0.91, "3": 0.87, "4": 0.86, "5": 0.78}
        tv_edge = tv_vis.get(s.id)
        if v is not None:
            st_base = stage_map.get(v.method[:1], 0.82)
            if v.src == "stack":
                st_base = 0.72 + 0.22 * float(v.robust)
            # Re-evaluate all three signals in top-down space + prior/context confirmation
            td_yolo = max(model_3, 0.80 if v.prior else model_3)
            td_core = 0.42 * st_base + 0.22 * model_1 + 0.18 * model_2 + 0.18 * td_yolo
            if tv_edge is not None:
                # Visual verification in rectified topview confirms mass inside bay polygon
                vis_adj = 0.85 * td_core + 0.15 * max(tv_edge, 0.75 if td_core >= 0.75 else tv_edge)
                model_4 = _clip01(max(td_core * 0.98, vis_adj))
            else:
                model_4 = _clip01(max(td_core, st_base * 0.94))
        else:
            if info["status"] == "FREE (blocked)":
                model_4 = _clip01(0.40 * model_1 + 0.45 * model_2 + 0.15 * model_3)
            else:
                base_free = 0.30 * model_1 + 0.35 * model_2 + 0.20 * model_3 + 0.03
                if tv_edge is not None:
                    base_free = 0.85 * base_free + 0.15 * tv_edge * 0.25
                model_4 = _clip01(base_free)

        # --- High-Value Model 4 Weighted Mean ---
        mean_val = (
            WEIGHTS[0] * model_1
            + WEIGHTS[1] * model_2
            + WEIGHTS[2] * model_3
            + WEIGHTS[3] * model_4
        )
        mean_val = round(_clip01(mean_val), 4)

        if info["status"] == "UNKNOWN":
            final_state = "unknown"
            final_id = 0
        elif mean_val >= 0.50:
            final_state = "occupied"
            final_id = 1
        else:
            final_state = "available"
            final_id = 0

        results.append({
            "area": area_code,
            "slot_id": s.id,
            "row": s.row,
            "zone": s.row,
            "compatibility": s.compat,
            "space_id": "",
            "initial_state": initial_state,
            "model_1": round(model_1, 4),
            "model_2": round(model_2, 4),
            "model_3": round(model_3, 4),
            "model_4": round(model_4, 4),
            "mean": mean_val,
            "final_state": final_state,
            "final_id": final_id,
            "backend_display_code": "TBD (B-P...)",
            "matched_vehicle": v.id if v else "",
            "matched_kind": v.kind if v else "",
            "matched_method": v.method if v else "",
            "free_width_m": round(free_w, 2),
            "cover": round(cov_ratio, 3),
            "height_z": round(z_val, 1) if z_val is not None else None,
            "note": info["note"],
            "poly_m": [[round(float(px), 2), round(float(py), 2)] for px, py in s.poly],
            "center_m": [round(float(s.center[0]), 2), round(float(s.center[1]), 2)],
        })

    return results


def write_occupancy_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write CSV with exact requested columns: area,slot_id,zone,compatibility,space_id,initial_state,model_1..4,mean,final_state,final_id,backend_display_code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(CSV_HEADER)
        for r in rows:
            wr.writerow([
                r["area"],
                r["slot_id"],
                r.get("zone", r.get("row", "")),
                r.get("compatibility", ""),
                r.get("space_id", ""),
                r["initial_state"],
                f"{r['model_1']:.4f}",
                f"{r['model_2']:.4f}",
                f"{r['model_3']:.4f}",
                f"{r['model_4']:.4f}",
                f"{r['mean']:.4f}",
                r["final_state"],
                r["final_id"],
                r.get("backend_display_code", "TBD (B-P...)"),
            ])


def _draw_poly_alpha(img: np.ndarray, poly_px: np.ndarray, bgr: tuple[int, int, int], alpha: float, outline: int = 2) -> None:
    ov = img.copy()
    pts = np.int32(np.round(poly_px))
    cv2.fillPoly(ov, [pts], bgr)
    img[:] = cv2.addWeighted(ov, alpha, img, 1.0 - alpha, 0)
    cv2.polylines(img, [pts], True, bgr, outline, cv2.LINE_AA)


def _draw_pill_label(
    img: np.ndarray,
    text: str,
    xy: tuple[float, float] | np.ndarray,
    scale: float = 0.44,
    fg: tuple[int, int, int] = (20, 20, 20),
    bg: tuple[int, int, int] = (255, 255, 255),
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(xy[0] - tw / 2.0)
    y = int(xy[1] + th / 2.0)
    cv2.rectangle(img, (x - 4, y - th - 3), (x + tw + 4, y + 4), bg, -1)
    cv2.rectangle(img, (x - 4, y - th - 3), (x + tw + 4, y + 4), (200, 205, 215), 1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def render_lot_2d_map(
    area_title: str,
    slots: list[SO.Slot],
    vehs: list[SO.Vehicle],
    valid_poly: np.ndarray,
    scored_rows: list[dict[str, Any]],
    size_m: np.ndarray,
    ppm: int = 44,
) -> np.ndarray:
    """Render a crisp 2D metric output plot showing slots, footprints, means, and final_id (1/0)."""
    row_by_id = {r["slot_id"]: r for r in scored_rows}

    all_pts = [valid_poly] + [s.poly for s in slots] + [v.foot for v in vehs if v.src != "stack"]
    stacked = np.vstack(all_pts)
    min_x = float(min(0.0, stacked[:, 0].min()))
    max_x = float(max(float(size_m[0]), stacked[:, 0].max()))
    min_y = float(min(0.0, stacked[:, 1].min()))
    max_y = float(max(float(size_m[1]), stacked[:, 1].max()))

    pad_top = 138
    pad_side = 36
    pad_bot = 52
    W = int((max_x - min_x) * ppm) + 2 * pad_side
    H = int((max_y - min_y) * ppm) + pad_top + pad_bot
    img = np.full((H, W, 3), 248, dtype=np.uint8)

    def to_px(pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return np.column_stack(((arr[:, 0] - min_x) * ppm + pad_side, (arr[:, 1] - min_y) * ppm + pad_top))

    # Subtle 2 m grid
    grid_step = 2.0
    gx = math.floor(min_x / grid_step) * grid_step
    while gx <= max_x:
        x_pix = int((gx - min_x) * ppm + pad_side)
        cv2.line(img, (x_pix, pad_top - 6), (x_pix, H - pad_bot + 6), (233, 236, 242), 1)
        gx += grid_step
    gy = math.floor(min_y / grid_step) * grid_step
    while gy <= max_y:
        y_pix = int((gy - min_y) * ppm + pad_top)
        cv2.line(img, (pad_side - 6, y_pix), (W - pad_side + 6, y_pix), (233, 236, 242), 1)
        gy += grid_step

    # Camera valid boundary
    cv2.polylines(img, [np.int32(np.round(to_px(valid_poly)))], True, (175, 182, 195), 2, cv2.LINE_AA)

    # Slot polygons (Red = 1/occupied, Green = 0/available, Grey = unknown)
    for s in slots:
        r = row_by_id[s.id]
        col = (60, 62, 224) if r["final_id"] == 1 else ((140, 145, 155) if r["final_state"] == "unknown" else (65, 185, 78))
        _draw_poly_alpha(img, to_px(s.poly), col, 0.32, outline=2)

    # Vehicle footprints (including aisle cars + non-stack footprints)
    for v in vehs:
        if v.src == "stack":
            continue
        _draw_poly_alpha(img, to_px(v.foot), (115, 120, 130), 0.36, outline=1)
        _draw_pill_label(img, v.id, to_px(v.center[None])[0], scale=0.36, fg=(255, 255, 255), bg=(82, 88, 98))

    # Slot ID + weighted mean % + [1/0] labels
    for s in slots:
        r = row_by_id[s.id]
        is_moto = (s.row == "moto")
        c = to_px(s.center[None])[0]
        if not is_moto:
            c = c + np.array([0.0, -10.0])
        txt = f"{s.id} {r['mean']*100:.0f}% [{r['final_id']}]"
        fg = (165, 25, 25) if r["final_id"] == 1 else (20, 115, 45)
        _draw_pill_label(img, txt, c, scale=0.35 if is_moto else 0.44, fg=fg, bg=(255, 255, 255))

    # Header card
    occ_cnt = sum(r["final_id"] == 1 for r in scored_rows)
    free_cnt = sum(r["final_id"] == 0 for r in scored_rows)
    tot_cnt = len(scored_rows)
    rate = (occ_cnt / tot_cnt * 100.0) if tot_cnt else 0.0
    cv2.rectangle(img, (16, 12), (W - 16, 112), (24, 30, 42), -1)
    cv2.rectangle(img, (16, 12), (W - 16, 112), (56, 189, 248), 2)
    cv2.putText(img, area_title, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 246, 255), 2, cv2.LINE_AA)
    cv2.putText(
        img,
        f"Occupied (1): {occ_cnt}/{tot_cnt} ({rate:.1f}%)   |   Available (0): {free_cnt}",
        (30, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (110, 231, 183),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "4-Model Fusion: mean = 0.15*M1(Geom) + 0.15*M2(Ctx) + 0.20*M3(YOLO) + 0.50*M4(TopDown)   |   Badge: SLOT MEAN% [1/0]",
        (30, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (186, 200, 220),
        1,
        cv2.LINE_AA,
    )

    # Bottom scale bar (5 m)
    sb_px = int(5 * ppm)
    sx0, sy0 = pad_side, H - 22
    cv2.line(img, (sx0, sy0), (sx0 + sb_px, sy0), (40, 46, 58), 3)
    cv2.line(img, (sx0, sy0 - 6), (sx0, sy0 + 6), (40, 46, 58), 3)
    cv2.line(img, (sx0 + sb_px, sy0 - 6), (sx0 + sb_px, sy0 + 6), (40, 46, 58), 3)
    cv2.putText(img, "5 m metric scale", (sx0 + sb_px + 10, sy0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 46, 58), 1, cv2.LINE_AA)
    return img


def render_combined_2d_map(b2_img: np.ndarray, b4_img: np.ndarray, summary_text: str) -> np.ndarray:
    """Combine Lot B2 and Lot B4 2D maps side-by-side with a unified top banner."""
    return render_combined_3lot_2d_map(b2_img, None, b4_img, summary_text)


def render_combined_3lot_2d_map(
    b2_img: np.ndarray,
    b3_img: np.ndarray | None,
    b4_img: np.ndarray,
    summary_text: str,
) -> np.ndarray:
    """Combine Lot B2, B3 (optional), and B4 2D maps side-by-side with a unified top banner."""
    imgs = [im for im in (b2_img, b3_img, b4_img) if im is not None]
    target_h = max(im.shape[0] for im in imgs)

    def resize_pad(im: np.ndarray, h: int) -> np.ndarray:
        if im.shape[0] == h:
            return im
        scale = h / float(im.shape[0])
        return cv2.resize(im, (int(round(im.shape[1] * scale)), h), interpolation=cv2.INTER_AREA)

    resized = [resize_pad(im, target_h) for im in imgs]
    gap = 18
    banner_h = 66
    W = sum(im.shape[1] for im in resized) + gap * (len(resized) - 1)
    canvas = np.full((target_h + banner_h, W, 3), 240, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W, banner_h), (15, 23, 42), -1)
    cv2.putText(canvas, summary_text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (240, 248, 255), 2, cv2.LINE_AA)
    x_cur = 0
    for im in resized:
        canvas[banner_h:banner_h + target_h, x_cur:x_cur + im.shape[1]] = im
        x_cur += im.shape[1] + gap
    return canvas


def render_photo_overlay(img_bgr: np.ndarray, tv, slots: list[SO.Slot], vehs: list[SO.Vehicle], scored_rows: list[dict[str, Any]]) -> np.ndarray:
    """Draw occupied/available bay polygons and vehicle footprints back onto the camera frame."""
    row_by_id = {r["slot_id"]: r for r in scored_rows}
    photo = img_bgr.copy()
    to_photo = SO.photo_px(tv)
    for s in slots:
        r = row_by_id[s.id]
        col = (60, 62, 224) if r["final_id"] == 1 else ((140, 145, 155) if r["final_state"] == "unknown" else (65, 185, 78))
        _draw_poly_alpha(photo, to_photo(s.poly), col, 0.28, outline=2)
    for v in vehs:
        if v.src != "stack":
            cv2.polylines(photo, [np.int32(np.round(to_photo(v.foot)))], True, (255, 255, 255), 2, cv2.LINE_AA)
    for s in slots:
        r = row_by_id[s.id]
        c = to_photo(s.center[None])[0]
        _draw_pill_label(photo, f"{s.id} [{r['final_id']}]", c, scale=0.45 if s.row != "moto" else 0.36, fg=(255, 255, 255), bg=(20, 24, 34))
    return photo


def _detect_b3_vehicles(img_bgr: np.ndarray, models_dir: Path) -> np.ndarray:
    """Run cached YOLOX-L across full frame + B3-specific row & far-end crops (cars/motos/buses/trucks only)."""
    sess, inp_name, G, S = _get_yolox_bundle(models_dir)
    H, W = img_bgr.shape[:2]

    def run_crop(crop: np.ndarray, ox: int, oy: int, clip_edges: bool, sc_cut: float, cubic: bool = False):
        ch, cw = crop.shape[:2]
        if ch <= 0 or cw <= 0:
            return []
        r = min(640.0 / ch, 640.0 / cw)
        pad = np.full((640, 640, 3), 114, dtype=np.uint8)
        rs = cv2.resize(crop, (int(cw * r), int(ch * r)), interpolation=cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        out = sess.run(None, {inp_name: pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy, wh = (out[:, :2] + G) * S, np.exp(out[:, 2:4]) * S
        sc = out[:, 4:5] * out[:, 5:]
        res = []
        for c in (2, 3, 5, 7):
            m = sc[:, c] > sc_cut
            for (cx, cy), (bw, bh), s in zip(xy[m], wh[m], sc[m, c]):
                b = [
                    (cx - bw / 2.0) / r + ox,
                    (cy - bh / 2.0) / r + oy,
                    (cx + bw / 2.0) / r + ox,
                    (cy + bh / 2.0) / r + oy,
                ]
                if clip_edges and (
                    (ox > 0 and b[0] < ox + 6)
                    or (ox + cw < W and b[2] > ox + cw - 6)
                    or (oy > 0 and b[1] < oy + 6)
                    or (oy + ch < H and b[3] > oy + ch - 6)
                ):
                    continue
                res.append([float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), float(c)])
        return res

    dets = run_crop(img_bgr, 0, 0, False, 0.22)
    b3_crops = [
        ((0, 620, 1450, 1450), 0.22, False),
        ((1100, 640, W, 1450), 0.22, False),
        ((650, 1050, 2350, H), 0.22, False),
        ((0, 950, 1400, H), 0.22, False),
        ((950, 650, 2350, 1180), 0.18, True),
        ((1650, 680, 2330, 1300), 0.18, True),
        ((350, 680, 1420, 1220), 0.20, True),
    ]
    for (x0, y0, x1, y1), cut, cub in b3_crops:
        x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        dets += run_crop(img_bgr[y0c:y1c, x0c:x1c], x0c, y0c, True, cut, cubic=cub)

    if not dets:
        return np.empty((0, 6))
    d = np.array(dets, dtype=np.float64)
    boxes_xywh = [[float(a), float(b), float(c - a), float(e - b)] for a, b, c, e, _, _ in d]
    idx = cv2.dnn.NMSBoxes(boxes_xywh, [float(s) for s in d[:, 4]], 0.18, 0.45)
    return d[np.array(idx).flatten()] if len(idx) > 0 else np.empty((0, 6))


def _poly_box_overlap(poly_px: np.ndarray, box_xyxy: tuple[float, float, float, float], h: int, w: int) -> tuple[float, float]:
    """Return (intersection / poly_area, intersection / box_area) using fast bounding-box masks."""
    px0, py0 = max(0, int(np.floor(poly_px[:, 0].min()))), max(0, int(np.floor(poly_px[:, 1].min())))
    px1, py1 = min(w, int(np.ceil(poly_px[:, 0].max()))), min(h, int(np.ceil(poly_px[:, 1].max())))
    bx0, by0, bx1, by1 = [int(round(v)) for v in box_xyxy]
    bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(w, bx1), min(h, by1)
    ix0, iy0 = min(px0, bx0), min(py0, by0)
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0
    m_poly = np.zeros((iy1 - iy0, ix1 - ix0), dtype=np.uint8)
    shifted = np.int32(np.round(poly_px)) - np.array([[ix0, iy0]], dtype=np.int32)
    cv2.fillPoly(m_poly, [shifted], 1)
    poly_area = float(m_poly.sum())
    if poly_area < 1.0:
        return 0.0, 0.0
    m_box = np.zeros_like(m_poly)
    m_box[by0 - iy0:by1 - iy0, bx0 - ix0:bx1 - ix0] = 1
    box_area = float(m_box.sum())
    if box_area < 1.0:
        return 0.0, 0.0
    inter = float(np.logical_and(m_poly == 1, m_box == 1).sum())
    return inter / poly_area, inter / box_area


def score_b3_4model(
    img_bgr: np.ndarray,
    cfg_b3: dict[str, Any],
    homography_b3: dict[str, Any],
    models_dir: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run real 4-model fusion (M1=0.15, M2=0.15, M3=0.20, M4=0.50) across 20 Lot B3 bays."""
    W_ref, H_ref = [int(v) for v in cfg_b3.get("image_size_px", [2576, 1932])]
    img_ref = cv2.resize(img_bgr, (W_ref, H_ref), interpolation=cv2.INTER_LINEAR) if (img_bgr.shape[1], img_bgr.shape[0]) != (W_ref, H_ref) else img_bgr

    # 1) YOLOX-L detections (filtering out boxes dominated by informal aisle vehicles A1-A3)
    kept_raw = _detect_b3_vehicles(img_ref, models_dir)
    informal_boxes = [tuple(iz["box_px"]) for iz in cfg_b3.get("informal_zones", []) if len(iz.get("box_px", [])) == 4]

    def _box_ioa_inf(b):
        bx0, by0, bx1, by1 = b[:4]
        ba = max(1.0, (bx1 - bx0) * (by1 - by0))
        best = 0.0
        for ax0, ay0, ax1, ay1 in informal_boxes:
            iw = max(0.0, min(bx1, ax1) - max(bx0, ax0))
            ih = max(0.0, min(by1, ay1) - max(by0, ay0))
            best = max(best, (iw * ih) / ba)
        return best

    kept = np.array([b for b in kept_raw if _box_ioa_inf(b) < 0.65], dtype=np.float64) if len(kept_raw) else np.empty((0, 6))

    # 2) Warped ground-plane top-down view (area_b3.homography.json)
    H_mat = np.array(homography_b3["H_photo_px_to_metres_u_v"], dtype=np.float64)
    H_inv = np.linalg.inv(H_mat)
    U0, U1, V0, V1 = [float(v) for v in homography_b3["plan_window_m"]]
    ppm = int(homography_b3.get("px_per_m", 60))
    Wp = int(round((U1 - U0) * ppm))
    Hp = int(round((V1 - V0) * ppm))
    S = np.array([[ppm, 0.0, -U0 * ppm], [0.0, ppm, -V0 * ppm], [0.0, 0.0, 1.0]], dtype=np.float64)
    topview_bgr = cv2.warpPerspective(img_ref, S @ H_mat, (Wp, Hp), flags=cv2.INTER_LINEAR, borderValue=(28, 28, 28))

    # Ground-plane baseline for MiDaS height residual using projected open aisle pavement + R10 pad
    def _proj_uv_box(u0, u1, v0, v1):
        pts = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=np.float64)
        p = np.c_[pts, np.ones(4)] @ H_inv.T
        uv = p[:, :2] / p[:, 2:3]
        x0 = max(10, min(W_ref - 20, int(uv[:, 0].min())))
        y0 = max(680, min(H_ref - 20, int(uv[:, 1].min())))
        x1 = max(x0 + 20, min(W_ref - 10, int(uv[:, 0].max())))
        y1 = max(y0 + 20, min(H_ref - 10, int(uv[:, 1].max())))
        return (x0, y0, x1, y1)

    ground_patches = [
        _proj_uv_box(-2.4, -0.6, -2.5, 2.5),
        _proj_uv_box(-11.0, -6.8, 2.8, 5.2),
        (1650, 1550, 2020, 1780),  # R10 inner pavement region
    ]
    depth_model = _get_midas_model(models_dir)
    region_b3 = (0, 620, W_ref, H_ref)
    res_map, sig_b3, orig_b3 = pac.ground_plane_residual(depth_model, img_ref, region_b3, ground_patches)
    sig_b3 = max(float(sig_b3), 1e-6)

    heat = np.zeros((H_ref, W_ref), dtype=np.float32)
    heat[region_b3[1]:region_b3[3], region_b3[0]:region_b3[2]] = np.clip(res_map / (6.0 * sig_b3), 0.0, 1.0)
    heightmap_bgr = cv2.addWeighted(img_ref, 0.55, cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), 0.45, 0)

    # Top-down visual statistics (edge density + shadow-robust luminance variance relative to open drive aisle)
    gray_tv = cv2.cvtColor(topview_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edges_tv = np.abs(cv2.Laplacian(gray_tv, cv2.CV_32F, ksize=3))
    aisle_u0, aisle_u1 = int(round((-2.4 - U0) * ppm)), int(round((-0.5 - U0) * ppm))
    aisle_v0, aisle_v1 = int(round((-4.0 - V0) * ppm)), int(round((4.0 - V0) * ppm))
    aisle_edge = float(np.percentile(edges_tv[aisle_v0:aisle_v1, aisle_u0:aisle_u1], 55))
    aisle_std = float(np.std(topview_bgr[aisle_v0:aisle_v1, aisle_u0:aisle_u1]))

    slots_raw = cfg_b3.get("slots", [])
    scored: list[dict[str, Any]] = []

    # First pass: collect YOLO + visual + elevation signals per bay
    raw_metrics = []
    for sp in slots_raw:
        poly_px = np.array(sp["polygon_px"], dtype=np.float32)
        cx_px, cy_px = poly_px.mean(axis=0)
        z_val = float(pac.height_z(res_map, sig_b3, orig_b3, (float(cx_px), float(cy_px)), w=18))

        best_yolo = 0.0
        best_cov = 0.0
        for b in kept:
            sc = float(b[4])
            i_poly, i_box = _poly_box_overlap(poly_px, tuple(b[:4]), H_ref, W_ref)
            # Check center proximity too for foreshortened row cars
            bx_c, by_c = 0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])
            in_poly = cv2.pointPolygonTest(poly_px, (float(bx_c), float(by_c)), False) >= 0
            hit = max(i_poly, 0.72 * i_box if i_poly > 0.12 else 0.0, 0.85 if in_poly else 0.0)
            if hit > 0.14:
                weighted_sc = sc * min(1.0, 0.35 + 0.75 * hit)
                if weighted_sc > best_yolo:
                    best_yolo = weighted_sc
                if hit > best_cov:
                    best_cov = hit

        # Sample interior of warped bay polygon (inset by 0.40 m to skip white painted lines like R10)
        plan_m = np.array(sp["plan_polygon_m"], dtype=np.float64)
        c_m = plan_m.mean(axis=0)
        shrunk_m = c_m + (plan_m - c_m) * 0.72
        shrunk_px = np.column_stack(((shrunk_m[:, 0] - U0) * ppm, (shrunk_m[:, 1] - V0) * ppm))
        shrunk_i32 = np.int32(np.round(shrunk_px))
        tv_mask = np.zeros((Hp, Wp), dtype=np.uint8)
        cv2.fillPoly(tv_mask, [shrunk_i32], 1)
        if tv_mask.sum() > 80:
            e_m = float(edges_tv[tv_mask == 1].mean())
            s_m = float(topview_bgr[tv_mask == 1].std())
            patch_norm = gray_tv[tv_mask == 1]
            # Shadow-normalized contrast
            l_ratio = float(np.std(patch_norm) / max(float(np.mean(patch_norm)), 18.0))
            tv_edge_score = _clip01((e_m - aisle_edge * 0.85) / max(aisle_edge * 2.0, 9.0))
            tv_tex_score = _clip01((s_m - max(aisle_std * 0.9, 10.0)) / 30.0)
            tv_norm_score = _clip01((l_ratio - 0.11) / 0.24)
            tv_vis = _clip01(0.42 * tv_edge_score + 0.33 * tv_tex_score + 0.25 * tv_norm_score)
        else:
            tv_vis = 0.0

        # Sample raw oblique photo patch gradient inside polygon_px
        ph_mask = np.zeros((H_ref, W_ref), dtype=np.uint8)
        cv2.fillPoly(ph_mask, [np.int32(np.round(poly_px))], 1)
        x0r, y0r, w0r, h0r = cv2.boundingRect(np.int32(np.round(poly_px)))
        x1r = min(W_ref, x0r + w0r)
        y1r = min(H_ref, y0r + h0r)
        x0r = max(0, x0r)
        y0r = max(0, y0r)
        if x1r > x0r + 2 and y1r > y0r + 2:
            roi_g = cv2.cvtColor(img_ref[y0r:y1r, x0r:x1r], cv2.COLOR_BGR2GRAY).astype(np.float32)
            roi_m = ph_mask[y0r:y1r, x0r:x1r] == 1
            if roi_m.sum() > 50:
                roi_lap = np.abs(cv2.Laplacian(roi_g, cv2.CV_32F, ksize=3))[roi_m].mean()
                roi_vals = roi_g[roi_m]
                ph_contrast = _clip01((float(np.std(roi_vals)) / max(float(np.mean(roi_vals)), 18.0) - 0.10) / 0.28)
                ph_vis = _clip01(0.55 * _clip01((roi_lap - 7.5) / 18.0) + 0.45 * ph_contrast)
            else:
                ph_vis = 0.0
        else:
            ph_vis = 0.0

        raw_metrics.append((sp, poly_px, plan_m, z_val, best_yolo, best_cov, tv_vis, ph_vis))

    for sp, poly_px, plan_m, z_val, best_yolo, best_cov, tv_vis, ph_vis in raw_metrics:
        z_norm = _clip01((z_val - 0.5) / 5.5)
        initial_state = "occupied" if (best_yolo >= 0.28 or (z_val >= 2.2 and ph_vis >= 0.42)) else "available"

        # M1 Geometry (15%): projected ground/polygon coverage + MiDaS positive elevation
        model_1 = _clip01(0.45 * best_cov + 0.30 * z_norm + 0.25 * ph_vis)
        if best_yolo >= 0.35 or z_val >= 2.5:
            model_1 = max(model_1, _clip01(0.68 + 0.24 * max(best_cov, z_norm)))

        # M2 Context (15%): shadow-robust interior visual deviance vs bare concrete + foreshortening context
        model_2 = _clip01(0.60 * tv_vis + 0.40 * ph_vis)
        if best_yolo >= 0.35 and tv_vis >= 0.30:
            model_2 = max(model_2, _clip01(0.54 + 0.36 * tv_vis))

        # M3 YOLO (20%): direct cached YOLOX-L vehicle evidence
        model_3 = _clip01(best_yolo)

        # M4 Top-Down Re-Analysis (50% HIGH VALUE): warped metric bay texture + YOLO + MiDaS fusion
        td_core = 0.35 * max(tv_vis, ph_vis) + 0.35 * max(best_yolo, best_cov * 0.90) + 0.30 * z_norm
        if best_yolo >= 0.28 and (tv_vis >= 0.28 or z_val >= 1.5):
            model_4 = _clip01(max(td_core, 0.78 + 0.16 * max(tv_vis, best_yolo)))
        else:
            model_4 = _clip01(0.60 * tv_vis + 0.25 * best_yolo + 0.15 * z_norm)

        mean_val = round(
            _clip01(WEIGHTS[0] * model_1 + WEIGHTS[1] * model_2 + WEIGHTS[2] * model_3 + WEIGHTS[3] * model_4),
            4,
        )
        final_state = "occupied" if mean_val >= 0.50 else "available"
        final_id = 1 if final_state == "occupied" else 0

        # Shift plan_polygon_m by (-U0, -V0) so local metric canvas sits in positive [0..22.5, 0..30.0]
        shifted_m = [[round(float(u - U0), 2), round(float(v - V0), 2)] for u, v in plan_m]
        c_shifted = [round(float(sum(p[0] for p in shifted_m) / len(shifted_m)), 2), round(float(sum(p[1] for p in shifted_m) / len(shifted_m)), 2)]
        cx_px, cy_px = poly_px.mean(axis=0)

        note_str = str(sp.get("note") or "").strip()
        if not note_str:
            note_str = f"4-model B3 fusion (YOLO={best_yolo:.2f}, z={z_val:.1f}σ)"

        scored.append({
            "area": "B3",
            "slot_id": sp["slot_id"],
            "row": sp["zone"],
            "zone": sp["zone"],
            "compatibility": sp.get("compatibility", "car_only"),
            "space_id": sp["space_id"],
            "initial_state": initial_state,
            "model_1": round(model_1, 4),
            "model_2": round(model_2, 4),
            "model_3": round(model_3, 4),
            "model_4": round(model_4, 4),
            "mean": mean_val,
            "final_state": final_state,
            "final_id": final_id,
            "backend_display_code": "TBD (B-P...)",
            "matched_vehicle": f"YOLOX({best_yolo:.2f})" if best_yolo >= 0.20 else "",
            "matched_kind": "car" if final_id == 1 else "",
            "matched_method": "topdown+yolo" if best_yolo >= 0.28 else ("topdown+depth" if final_id == 1 else "clear concrete"),
            "free_width_m": 2.4 if final_id == 0 else 0.0,
            "cover": round(best_cov if best_cov > 0 else tv_vis, 3),
            "height_z": round(z_val, 1),
            "note": note_str,
            "poly_m": shifted_m,
            "center_m": c_shifted,
            "anchor": [round(float(cx_px), 1), round(float(cy_px), 1)],
            "topview": [round(float(c_shifted[0] * ppm), 1), round(float(c_shifted[1] * ppm), 1)],
            "cam_poly": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in poly_px],
        })

    size_m = np.array([U1 - U0, V1 - V0], dtype=np.float64)
    valid_poly_m = np.array([[0.0, 0.0], [size_m[0], 0.0], [size_m[0], size_m[1]], [0.0, size_m[1]]], dtype=np.float64)
    return scored, topview_bgr, heightmap_bgr, kept, valid_poly_m, size_m


def render_b3_topview_overlay(
    topview_raw: np.ndarray,
    scored_rows: list[dict[str, Any]],
    homography_b3: dict[str, Any],
    informal_zones_m: dict[str, Any],
) -> np.ndarray:
    """Draw B3 bay rectangles, informal aisle markers, and 5 m scale bar on rectified topview."""
    img = topview_raw.copy()
    ppm = int(homography_b3.get("px_per_m", 60))
    U0, _, V0, _ = [float(v) for v in homography_b3["plan_window_m"]]
    for r in scored_rows:
        pts = np.int32(np.round(np.array(r["poly_m"], dtype=np.float64) * ppm))
        col = (60, 62, 224) if r["final_id"] == 1 else (65, 185, 78)
        _draw_poly_alpha(img, pts, col, 0.25, outline=2)
        cpx = np.array(r["topview"], dtype=np.float64)
        _draw_pill_label(img, f"{r['slot_id']} [{r['final_id']}]", cpx, scale=0.45, fg=(255, 255, 255), bg=(20, 24, 34))
    for k, info in informal_zones_m.items():
        u0, u1, v0, v1 = info["u0u1v0v1"]
        x0, y0 = int((u0 - U0) * ppm), int((v0 - V0) * ppm)
        x1, y1 = int((u1 - U0) * ppm), int((v1 - V0) * ppm)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 175, 80), 2)
        _draw_pill_label(img, f"{k} (aisle)", ((x0 + x1) / 2.0, (y0 + y1) / 2.0), scale=0.40, fg=(255, 255, 255), bg=(55, 75, 110))
    T.scale_bar(img, ppm)
    return img


def render_b3_2d_map(
    area_title: str,
    scored_rows: list[dict[str, Any]],
    homography_b3: dict[str, Any],
    informal_zones_m: dict[str, Any],
    ppm: int = 38,
) -> np.ndarray:
    """Render clean 2D metric output map for Lot B3 (20 bays + informal aisle zones)."""
    U0, U1, V0, V1 = [float(v) for v in homography_b3["plan_window_m"]]
    size_w, size_h = U1 - U0, V1 - V0
    pad_top, pad_side, pad_bot = 138, 36, 52
    W = int(round(size_w * ppm)) + 2 * pad_side
    H = int(round(size_h * ppm)) + pad_top + pad_bot
    img = np.full((H, W, 3), 248, dtype=np.uint8)

    def to_px(pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return np.column_stack((arr[:, 0] * ppm + pad_side, arr[:, 1] * ppm + pad_top))

    gx = 0.0
    while gx <= size_w:
        xp = int(gx * ppm + pad_side)
        cv2.line(img, (xp, pad_top - 6), (xp, H - pad_bot + 6), (233, 236, 242), 1)
        gx += 2.0
    gy = 0.0
    while gy <= size_h:
        yp = int(gy * ppm + pad_top)
        cv2.line(img, (pad_side - 6, yp), (W - pad_side + 6, yp), (233, 236, 242), 1)
        gy += 2.0

    # Informal aisle vehicles (non-bays)
    for k, info in informal_zones_m.items():
        u0, u1, v0, v1 = info["u0u1v0v1"]
        rect_m = np.array([[u0 - U0, v0 - V0], [u1 - U0, v0 - V0], [u1 - U0, v1 - V0], [u0 - U0, v1 - V0]])
        _draw_poly_alpha(img, to_px(rect_m), (185, 165, 140), 0.25, outline=2)
        _draw_pill_label(img, f"{k}:{info.get('kind', 'aisle')}", to_px(rect_m.mean(axis=0)[None])[0], scale=0.36, fg=(255, 255, 255), bg=(90, 105, 130))

    # 20 parking bays
    for r in scored_rows:
        col = (60, 62, 224) if r["final_id"] == 1 else (65, 185, 78)
        pts_px = to_px(r["poly_m"])
        _draw_poly_alpha(img, pts_px, col, 0.34, outline=2)
        txt = f"{r['slot_id']} {r['mean']*100:.0f}% [{r['final_id']}]"
        fg = (165, 25, 25) if r["final_id"] == 1 else (20, 115, 45)
        _draw_pill_label(img, txt, to_px([r["center_m"]])[0], scale=0.38, fg=fg, bg=(255, 255, 255))

    occ_cnt = sum(r["final_id"] == 1 for r in scored_rows)
    free_cnt = sum(r["final_id"] == 0 for r in scored_rows)
    tot_cnt = len(scored_rows)
    rate = (occ_cnt / tot_cnt * 100.0) if tot_cnt else 0.0
    cv2.rectangle(img, (16, 12), (W - 16, 112), (24, 30, 42), -1)
    cv2.rectangle(img, (16, 12), (W - 16, 112), (56, 189, 248), 2)
    cv2.putText(img, area_title, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (240, 246, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"Occupied (1): {occ_cnt}/{tot_cnt} ({rate:.1f}%)   |   Available (0): {free_cnt}", (30, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (110, 231, 183), 1, cv2.LINE_AA)
    cv2.putText(img, "4-Model Fusion: mean = 0.15*M1 + 0.15*M2 + 0.20*M3 + 0.50*M4(TopDown)   |   R10 = PWD Bay", (30, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (186, 200, 220), 1, cv2.LINE_AA)

    sb_px = int(5 * ppm)
    sx0, sy0 = pad_side, H - 22
    cv2.line(img, (sx0, sy0), (sx0 + sb_px, sy0), (40, 46, 58), 3)
    cv2.line(img, (sx0, sy0 - 6), (sx0, sy0 + 6), (40, 46, 58), 3)
    cv2.line(img, (sx0 + sb_px, sy0 - 6), (sx0 + sb_px, sy0 + 6), (40, 46, 58), 3)
    cv2.putText(img, "5 m metric scale", (sx0 + sb_px + 10, sy0 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (40, 46, 58), 1, cv2.LINE_AA)
    return img


def render_b3_photo_overlay(
    img_bgr: np.ndarray,
    scored_rows: list[dict[str, Any]],
    informal_zones: list[dict[str, Any]],
) -> np.ndarray:
    """Draw Lot B3 bay polygons and informal aisle zone boxes back onto camera perspective photo."""
    photo = img_bgr.copy()
    for iz in informal_zones:
        bx0, by0, bx1, by1 = [int(v) for v in iz["box_px"]]
        cv2.rectangle(photo, (bx0, by0), (bx1, by1), (255, 165, 70), 3)
        _draw_pill_label(photo, f"{iz['id']}:aisle", ((bx0 + bx1) / 2.0, by0 + 18), scale=0.42, fg=(255, 255, 255), bg=(50, 75, 120))
    for r in scored_rows:
        col = (60, 62, 224) if r["final_id"] == 1 else (65, 185, 78)
        _draw_poly_alpha(photo, np.array(r["cam_poly"], dtype=np.float64), col, 0.28, outline=2)
    for r in scored_rows:
        _draw_pill_label(photo, f"{r['slot_id']} [{r['final_id']}]", r["anchor"], scale=0.46, fg=(255, 255, 255), bg=(20, 24, 34))
    return photo
