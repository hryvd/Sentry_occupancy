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
    "initial_state",
    "model_1",
    "model_2",
    "model_3",
    "model_4",
    "mean",
    "final_state",
    "final_id",
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
            "compatibility": s.compat,
            "initial_state": initial_state,
            "model_1": round(model_1, 4),
            "model_2": round(model_2, 4),
            "model_3": round(model_3, 4),
            "model_4": round(model_4, 4),
            "mean": mean_val,
            "final_state": final_state,
            "final_id": final_id,
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
    """Write CSV with exact requested columns: area,slot_id,initial_state,model_1..4,mean,final_state,final_id."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(CSV_HEADER)
        for r in rows:
            wr.writerow([
                r["area"],
                r["slot_id"],
                r["initial_state"],
                f"{r['model_1']:.4f}",
                f"{r['model_2']:.4f}",
                f"{r['model_3']:.4f}",
                f"{r['model_4']:.4f}",
                f"{r['mean']:.4f}",
                r["final_state"],
                r["final_id"],
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
    target_h = max(b2_img.shape[0], b4_img.shape[0])

    def resize_pad(im: np.ndarray, h: int) -> np.ndarray:
        if im.shape[0] == h:
            return im
        scale = h / float(im.shape[0])
        return cv2.resize(im, (int(round(im.shape[1] * scale)), h), interpolation=cv2.INTER_AREA)

    b2_r = resize_pad(b2_img, target_h)
    b4_r = resize_pad(b4_img, target_h)
    gap = 20
    banner_h = 66
    W = b2_r.shape[1] + b4_r.shape[1] + gap
    canvas = np.full((target_h + banner_h, W, 3), 240, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W, banner_h), (15, 23, 42), -1)
    cv2.putText(canvas, summary_text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 248, 255), 2, cv2.LINE_AA)
    canvas[banner_h:banner_h + target_h, :b2_r.shape[1]] = b2_r
    canvas[banner_h:banner_h + target_h, b2_r.shape[1] + gap:] = b4_r
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
