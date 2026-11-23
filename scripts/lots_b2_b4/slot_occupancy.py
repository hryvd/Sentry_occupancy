#!/usr/bin/env python3
"""
scripts/lots_b2_b4/slot_occupancy.py
====================================
Metric bird's-eye view occupancy engine for Lot B2 (22 bays) and Lot B4 (27 bays).
Supports both calibrated baseline detections and live cached ONNX YOLOX-L detections.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

import cv2
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
AREA_B2_DIR = THIS_DIR.parent / "area_b2"
for _p in (str(THIS_DIR), str(AREA_B2_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import parking_lot2 as P2  # noqa: E402
import topview_slots as T  # noqa: E402

PPM = T.PPM
MIN_CAR_GAP_M = 2.2
KIND = {
    "car": (4.5, 1.85),
    "truck": (5.4, 1.95),
    "bus": (6.0, 2.1),
    "moto": (2.0, 0.8),
    "cart": (3.0, 1.5),
    "tricycle": (3.0, 1.6),
}
COCO_KIND = {2: "car", 3: "moto", 5: "bus", 7: "truck"}
STATUS_COLOR = {
    "OCCUPIED": (60, 60, 230),
    "FREE": (70, 190, 70),
    "FREE (blocked)": (0, 165, 255),
    "UNKNOWN": (150, 150, 150),
}


def area(p) -> float:
    """Shoelace polygon area in square metres."""
    p = np.asarray(p, float)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def convex_ccw(p) -> np.ndarray:
    """Convex hull in CCW float32 format."""
    p = np.asarray(p, np.float32)
    return p if area(p) == 0 else cv2.convexHull(p)[:, 0, :]


def overlap(a, b) -> float:
    """Intersection area of two convex polygons (metres²)."""
    ca, cb = convex_ccw(a), convex_ccw(b)
    if len(ca) < 3 or len(cb) < 3:
        return 0.0
    return float(cv2.intersectConvexConvex(ca, cb)[0])


def inside(poly, pt) -> bool:
    """Point-in-polygon test."""
    return cv2.pointPolygonTest(convex_ccw(poly).astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0


class Slot:
    """Metric 2D parking bay polygon."""

    def __init__(self, sid: str, row: str, poly, inferred: bool = False) -> None:
        self.id = sid
        self.row = row
        self.poly = np.asarray(poly, float)
        self.inferred = inferred
        self.center = self.poly.mean(0)
        self.t = float(self.center[1])
        self.t0 = float(self.poly[:, 1].min())
        self.t1 = float(self.poly[:, 1].max())
        self.area = area(self.poly)
        self.compat = "motorcycle_only" if row == "moto" else "car_only"


class Vehicle:
    """Detected or hand-annotated vehicle with ground tyre-line footprint."""

    def __init__(self, vid: str, kind: str, box, src: str, score: float = 1.0) -> None:
        self.id = vid
        self.kind = kind
        self.box = tuple(box)
        self.src = src
        self.score = float(score)
        self.foot: np.ndarray | None = None
        self.center: np.ndarray | None = None
        self.t: float | None = None
        self.slot: str | None = None
        self.method = ""
        self.zone = "aisle"
        self.prior = False
        self.robust = 1.0


def footprint(v: Vehicle, tv, axis=None):
    """Build metric ground footprint (4x2, metres) from tyre line at bottom of box."""
    x1, _, x2, y2 = v.box
    S = tv.to_metres([(x1, y2), (x2, y2)])
    d = S[1] - S[0]
    sl = max(float(np.linalg.norm(d)), 1e-4)
    L, W = KIND[v.kind]
    u = d / sl
    side_on = sl >= 0.62 * L or v.kind == "moto" or axis is not None
    if axis is not None:
        u = np.asarray(axis, float)
        if side_on:
            sl = min(sl, L * 1.12)
    n = np.array([u[1], -u[0]])
    if n[1] > 0:
        n = -n
    mid = S.mean(0)
    if side_on:
        length, depth = min(sl, L * 1.12), W
    else:
        length, depth = min(sl, W * 1.25), L
    p1, p2 = mid - u * length / 2.0, mid + u * length / 2.0
    return np.array([p1, p2, p2 + n * depth, p1 + n * depth]), side_on


# Calibrated baseline detections for Lot B4 (27 bays: L1-L8, R1-R10, MB1-MB9)
LOT2_DETECTIONS = [
    (660, 182, 1018, 377, 7, 0.79), (1017, 227, 1200, 453, 2, 0.64), (621, 372, 991, 568, 2, 0.74),
    (1038, 469, 1205, 709, 2, 0.96), (605, 526, 977, 702, 2, 0.95), (597, 667, 997, 862, 2, 0.95),
    (539, 837, 1015, 1053, 2, 0.92), (534, 1036, 1083, 1228, 2, 0.84), (494, 1332, 1124, 1528, 2, 0.45),
    (1231, 135, 1452, 248, 2, 0.74), (1286, 221, 1542, 329, 2, 0.91), (1323, 308, 1578, 410, 2, 0.94),
    (1341, 392, 1632, 497, 2, 0.93), (1365, 472, 1672, 615, 2, 0.95), (1412, 592, 1747, 724, 2, 0.94),
    (1422, 702, 1831, 878, 2, 0.93), (1506, 862, 1938, 1062, 2, 0.92), (1258, 864, 1733, 1517, 5, 0.75),
    (1584, 1055, 2050, 1278, 2, 0.92), (1677, 1306, 2046, 1485, 2, 0.39),
]
LOT2_MANUAL = [
    ("cart", (690, 290, 935, 425)), ("tricycle", (1040, 860, 1235, 1115)),
    ("moto", (301, 741, 512, 866)), ("moto", (340, 821, 549, 929)), ("moto", (287, 909, 519, 1010)),
    ("moto", (313, 1009, 500, 1078)), ("moto", (261, 1078, 478, 1145)), ("moto", (200, 1140, 437, 1254)),
    ("moto", (156, 1255, 415, 1354)), ("moto", (130, 1378, 383, 1457)), ("moto", (278, 1457, 357, 1496)),
    ("moto", (64, 855, 224, 1025)),
]


def lot2_slots() -> list[Slot]:
    """Build the 27 painted bays of Lot B4 in metres."""
    slots: list[Slot] = []
    for k in range(8):
        y0 = 9.0 + 2.4 * k
        slots.append(Slot(f"L{k + 1}", "left", [(4.4, y0), (9.2, y0), (9.2, y0 + 2.4), (4.4, y0 + 2.4)]))
    th = np.radians(-10.0)
    v = np.array([np.cos(th), np.sin(th)]) * 5.0
    for k in range(10):
        y0, y1 = 3.4 + 2.4 * k, 5.8 + 2.4 * k
        p1 = np.array([14.0 + 0.045 * (y0 - 5.8), y0])
        p2 = np.array([14.0 + 0.045 * (y1 - 5.8), y1])
        slots.append(Slot(f"R{k + 1}", "right", [p1, p2, p2 + v, p1 + v]))
    for k in range(9):
        y0 = 20.01 + 0.82 * k
        slots.append(Slot(f"MB{k + 1}", "moto", [(1.9, y0), (4.2, y0), (4.2, y0 + 0.82), (1.9, y0 + 0.82)]))
    return slots


def lot2_zone(c) -> str:
    x, y = c
    if x < 4.3 and y > 18.0:
        return "moto"
    if x < 9.9:
        return "left"
    if x > 13.2:
        return "right"
    return "aisle"


ROW_AXIS = {
    "left": (1.0, 0.0),
    "right": (float(np.cos(np.radians(-10))), float(np.sin(np.radians(-10)))),
    "moto": (1.0, 0.0),
}

LOT1_DETECTIONS = [
    ("left", (-2, 764, 418, 1034), 0.95), ("left", (133, 683, 561, 861), 0.93), ("left", (446, 578, 715, 693), 0.93),
    ("left", (575, 552, 782, 648), 0.93), ("left", (641, 522, 794, 603), 0.90), ("left", (685, 508, 844, 582), 0.83),
    ("left", (720, 492, 883, 556), 0.85), ("left", (758, 474, 889, 528), 0.84), ("left", (789, 456, 920, 516), 0.90),
    ("left", (867, 454, 956, 503), 0.86),
    ("right", (1466, 907, 2048, 1200), 0.96), ("right", (1397, 762, 1923, 953), 0.93), ("right", (1387, 677, 1763, 805), 0.93),
    ("right", (1318, 598, 1701, 739), 0.92), ("right", (1295, 558, 1555, 638), 0.90), ("right", (1279, 503, 1527, 596), 0.90),
    ("aisle", (514, 852, 1032, 1091), 0.94), ("aisle", (1017, 682, 1206, 937), 0.94), ("aisle", (584, 658, 805, 878), 0.94),
    ("aisle", (804, 679, 1026, 874), 0.93), ("aisle", (926, 574, 1058, 684), 0.93), ("aisle", (995, 516, 1089, 602), 0.91),
    ("aisle", (1041, 500, 1099, 550), 0.81),
]
LOT1_STACKS = [dict(id="S1", row="right", box=(1266, 461, 1420, 540), score=0.83, x_range=(1300, 1425), y_range=(452, 512))]
LOT1_MOTO = (22, 1000, 240, 1162)
LOT1_ROW_X = {"left": (3.6, 10.2), "right": (17.0, 22.2)}
LOT1_EXPECTED = 11
LOT1_CONFIRMED_FREE = ["L3"]
LOT1_SITE_REPORT = True


def roof_lines(img, x_range, y_range, n_bands=4, gap=3, min_votes=2, merge=4, prom=14):
    """Count stacked cars behind van via horizontal roof/window intensity peaks."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    x0, x1 = x_range
    y0, y1 = y_range
    edges = np.linspace(x0, x1, n_bands + 1).astype(int)
    peaks = []
    for b in range(n_bands):
        prof = g[y0:y1, edges[b]:edges[b + 1]].mean(1)
        prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0.9).ravel()
        for i in range(2, len(prof) - 2):
            if prof[i] > prof[i - 1] and prof[i] >= prof[i + 1] and prof[i] - prof[max(0, i - 4):i + 5].min() > prom:
                peaks.append((y0 + i, b))
    peaks.sort()
    clusters = []
    for y, b in peaks:
        if clusters and y - clusters[-1][0][-1] <= gap:
            clusters[-1][0].append(y)
            clusters[-1][1].add(b)
        else:
            clusters.append(([y], {b}))
    lines = [float(np.mean(c[0])) for c in clusters if len(c[1]) >= min_votes]
    merged = []
    for y in lines:
        if merged and y - merged[-1] <= merge:
            merged[-1] = (merged[-1] + y) / 2.0
        else:
            merged.append(y)
    return merged


def roof_count_robustness(img, x_range, y_range):
    """Share of roof-line sensitivity settings finding at least n stacked vehicles."""
    counts = []
    for dx in (0, -5, 5, -10, 10):
        xr = (x_range[0] + dx, x_range[1] + dx)
        for nb in (3, 4, 5):
            for gap in (2, 3):
                for mv in (2, 3):
                    for merge in (3, 4):
                        for prom in (12, 14, 16):
                            if mv <= nb:
                                counts.append(len(roof_lines(img, xr, y_range, nb, gap, mv, merge, prom)))
    arr = np.array(counts)
    return {n: float((arr >= n).mean()) for n in range(1, int(arr.max()) + 1)} if arr.size else {}


class Stack:
    """Stacked nose-to-tail rear-right car group."""

    def __init__(self, cfg: dict, img: np.ndarray) -> None:
        self.id = cfg["id"]
        self.row = cfg["row"]
        self.box = cfg["box"]
        self.score = cfg["score"]
        self.roofs = sorted(roof_lines(img, cfg["x_range"], cfg["y_range"]), reverse=True)
        rb = roof_count_robustness(img, cfg["x_range"], cfg["y_range"])
        self.robust = [rb.get(k + 1, 0.0) for k in range(len(self.roofs))]


def mark_prior(vehs, anchors) -> None:
    """Mark whether an earlier reference anchor lies inside each vehicle box."""
    for v in vehs:
        v.prior = any(a is not None and v.box[0] <= a[0] <= v.box[2] and v.box[1] <= a[1] <= v.box[3] for a in anchors)


def lot1_zone(c) -> str:
    x = c[0]
    return "left" if x < 9.9 else "right" if x > 16.6 else "aisle"


def infer_row(vs, row: str, n_expected: int, tag: str) -> list[Slot]:
    """Infer 11 unlined bays per row along Lot B2 from tyre lines & inter-vehicle gaps."""
    vs = sorted(vs, key=lambda v: -v.foot[:, 1].max())
    south = [v.foot[:, 1].max() + 0.3 for v in vs] if vs else [19.0]
    d = [south[i] - south[i + 1] for i in range(len(south) - 1)]
    p = float(np.clip(np.median([x for x in d if 1.4 <= x <= 3.2] or [2.1]), 2.0, 2.4))
    x0, x1 = LOT1_ROW_X[row]
    bays: list[Slot] = []
    cursor, k = south[0], 1

    def add(top: float, bottom: float) -> None:
        nonlocal k
        poly = [(x0, top), (x1, top), (x1, bottom), (x0, bottom)]
        bays.append(Slot(f"{tag}{k}", row, poly, inferred=True))
        k += 1

    for i, _ in enumerate(vs):
        n_fill = max(0, int(round((south[i] - south[i + 1]) / p)) - 1) if i + 1 < len(vs) else 0
        if i + 1 < len(vs):
            w = p if n_fill else min(max(south[i] - south[i + 1], 1.8), 2.6)
        else:
            w = p
        add(south[i] - w, south[i])
        cursor = south[i] - w
        for _ in range(n_fill):
            add(cursor - p, cursor)
            cursor -= p
    while k <= n_expected:
        add(cursor - p, cursor)
        cursor -= p
    return bays


def build_vehicles(boxes_kind, tv, zone_fn) -> list[Vehicle]:
    """Project tyre-line segments to metric footprints and assign initial row zones."""
    vehs = []
    for i, (kind, box, src, score) in enumerate(boxes_kind, 1):
        v = Vehicle(f"V{i}", kind, box, src, score)
        v.foot, _ = footprint(v, tv)
        v.center = v.foot.mean(0)
        v.zone = zone_fn(v.center)
        axis = ROW_AXIS.get(v.zone)
        if axis is not None:
            v.foot, _ = footprint(v, tv, axis)
            v.center = v.foot.mean(0)
            v.zone = zone_fn(v.center)
        v.t = float(v.center[1])
        vehs.append(v)
    return vehs


def _box_inside(inner, outer, pad: float = 16.0) -> bool:
    return (
        inner[0] >= outer[0] - pad
        and inner[1] >= outer[1] - pad
        and inner[2] <= outer[2] + pad
        and inner[3] <= outer[3] + pad
    )


def run_lot1(
    img: np.ndarray,
    live_boxes: list[tuple[str, tuple[float, float, float, float], str, float]] | None = None,
    fixed_slots: list[Slot] | None = None,
    enable_stack: bool = True,
):
    """Run Lot B2 (22 bays) using live YOLOX boxes or calibrated baseline detections."""
    h, w = img.shape[:2]
    tv = T.TopView(w, h)
    s1_box = LOT1_STACKS[0]["box"]

    if live_boxes is not None:
        filtered = []
        s1_hits = []
        for kind, box, src, sc in live_boxes:
            if enable_stack and _box_inside(box, s1_box, pad=22.0):
                s1_hits.append(sc)
                continue
            filtered.append((kind, box, src, sc))
        vehs = build_vehicles(filtered, tv, lot1_zone)
    else:
        boxes = [("car", b, "yolox", sc) for _, b, sc in LOT1_DETECTIONS] + [("moto", LOT1_MOTO, "yolox", 0.90)]
        vehs = build_vehicles(boxes, tv, lot1_zone)
        s1_hits = [LOT1_STACKS[0]["score"]]

    mark_prior(vehs, [sl[3] for sl in T.SLOTS])

    if enable_stack and s1_hits:
        st_cfg = dict(LOT1_STACKS[0])
        st_cfg["score"] = float(max(s1_hits))
        st = Stack(st_cfg, img)
        st.n_prior = sum(
            1 for sl in T.SLOTS
            if sl[3] and st.box[0] <= sl[3][0] <= st.box[2] and st.box[1] <= sl[3][1] <= st.box[3]
        )
        tv.stacks = [st]
    else:
        tv.stacks = []

    for v in vehs:
        if v.kind == "moto":
            v.zone = "aisle"

    valid = tv.to_metres(T.POLY)
    if fixed_slots is not None:
        slots = [Slot(s.id, s.row, s.poly.copy(), inferred=s.inferred) for s in fixed_slots]
    else:
        slots = []
        for row, tag in (("left", "L"), ("right", "R")):
            slots += infer_row([v for v in vehs if v.zone == row], row, LOT1_EXPECTED, tag)
    return tv, slots, vehs, valid


def run_lot2(
    img_shape_or_img,
    live_boxes: list[tuple[str, tuple[float, float, float, float], str, float]] | None = None,
):
    """Run Lot B4 (27 bays: L1-L8, R1-R10, MB1-MB9)."""
    if isinstance(img_shape_or_img, np.ndarray):
        h, w = img_shape_or_img.shape[:2]
    else:
        h, w = int(img_shape_or_img[0]), int(img_shape_or_img[1])
    tv = P2.fit_camera(w, h)

    if live_boxes is not None:
        boxes = list(live_boxes)
    else:
        boxes = [(COCO_KIND[c], (x1, y1, x2, y2), "yolox", sc) for x1, y1, x2, y2, c, sc in LOT2_DETECTIONS]
        for kind, box in LOT2_MANUAL:
            boxes.append((kind, box, "marked by hand", 1.0))

    slots = lot2_slots()
    vehs = build_vehicles(boxes, tv, lot2_zone)
    mark_prior(vehs, [sl[3] for sl in P2.SLOTS])
    valid = tv.to_metres(P2.POLY)
    return tv, slots, vehs, valid


def in_row_x(v: Vehicle, ss: list[Slot], min_frac: float = 0.4) -> bool:
    """Check whether >= min_frac of vehicle footprint spans the row corridor."""
    lo = min(s.poly[:, 0].min() for s in ss)
    hi = max(s.poly[:, 0].max() for s in ss)
    a, b = float(v.foot[:, 0].min()), float(v.foot[:, 0].max())
    return (min(b, hi) - max(a, lo)) >= min_frac * max(b - a, 1e-4)


def order_match(vs: list[Vehicle], ss: list[Slot], R: dict) -> dict[str, str]:
    """Order-preserving DP matching along row with global shift compensation."""
    nv, ns = len(vs), len(ss)
    if nv == 0 or ns == 0:
        return {}
    pitch = float(np.median([s.t1 - s.t0 for s in ss]))
    skip_v = 1.4
    best = (np.inf, {})
    for off in np.arange(-0.6, 0.61, 0.05):
        cost = np.full((nv, ns), 1e3)
        for a, v in enumerate(vs):
            for b, s in enumerate(ss):
                shift = abs(v.t - s.t - off) / max(pitch, 1e-4)
                if shift <= 1.6 or R[v.id].get(s.id, 0) > 0.05:
                    cost[a, b] = shift - R[v.id].get(s.id, 0)
        dp = np.full((nv + 1, ns + 1), np.inf)
        bk: dict[tuple[int, int], tuple[str, int, int]] = {}
        dp[0, :] = 0
        for a in range(1, nv + 1):
            dp[a, 0] = dp[a - 1, 0] + skip_v
            bk[(a, 0)] = ("v", a - 1, 0)
            for b in range(1, ns + 1):
                opts = [
                    (dp[a - 1, b] + skip_v, ("v", a - 1, b)),
                    (dp[a, b - 1], ("s", a, b - 1)),
                    (dp[a - 1, b - 1] + cost[a - 1, b - 1], ("m", a - 1, b - 1)),
                ]
                dp[a, b], bk[(a, b)] = min(opts, key=lambda o: o[0])
        if dp[nv, ns] < best[0]:
            out, a, b = {}, nv, ns
            while a > 0:
                kind, a2, b2 = bk.get((a, b), ("v", a - 1, b))
                if kind == "m" and cost[a - 1, b - 1] < 1e3:
                    out[vs[a - 1].id] = ss[b - 1].id
                a, b = a2, b2
            best = (float(dp[nv, ns]), out)
    return best[1]


def assign(slots: list[Slot], vehs: list[Vehicle], stacks=()):
    """5-stage metric top-down slot assignment."""
    byid = {s.id: s for s in slots}
    R = {v.id: {s.id: overlap(v.foot, s.poly) / max(area(v.foot), 1e-4) for s in slots} for v in vehs}
    held: dict[str, list[Vehicle]] = {}

    def claim(v: Vehicle, sid: str, how: str) -> None:
        v.slot, v.method = sid, how
        held.setdefault(sid, []).append(v)

    def release_conflicts() -> None:
        for sid, vs in list(held.items()):
            if len(vs) > 1:
                for v in vs:
                    v.slot, v.method = None, ""
                del held[sid]

    for v in vehs:
        for sid in R[v.id]:
            if (v.kind == "moto") != (byid[sid].row == "moto"):
                R[v.id][sid] = 0.0

    # Stage 1: clear >= 60% footprint overlap
    for v in vehs:
        r = sorted(R[v.id].items(), key=lambda kv: -kv[1])
        if r and r[0][1] >= 0.60 and (len(r) < 2 or r[1][1] < 0.25):
            claim(v, r[0][0], "1 overlap")
    release_conflicts()

    # Stage 2: footprint centre containment
    for v in vehs:
        if v.slot or sum(R[v.id].values()) < 0.2:
            continue
        for s in slots:
            if inside(s.poly, v.center) and s.id not in held and R[v.id][s.id] >= 0.35:
                claim(v, s.id, "2 centre")
                break
    release_conflicts()

    # Stage 3: order-preserving row DP
    for row in sorted({s.row for s in slots}):
        row_slots = [s for s in slots if s.row == row]
        rv = sorted([v for v in vehs if not v.slot and v.zone == row and in_row_x(v, row_slots)], key=lambda v: v.t)
        rs = sorted([s for s in row_slots if s.id not in held], key=lambda s: s.t)
        for vid, sid in order_match(rv, rs, R).items():
            claim(next(v for v in vehs if v.id == vid), sid, "3 order")

    # Stage 4: whole-row re-fit
    for row in sorted({s.row for s in slots}):
        row_slots = [s for s in slots if s.row == row]
        rv_all = [v for v in vehs if v.zone == row and sum(R[v.id].values()) >= 0.2 and in_row_x(v, row_slots)]
        stuck = [v for v in rv_all if not v.slot]
        free_bays = [s for s in slots if s.row == row and s.id not in held]
        if stuck and free_bays:
            for v in rv_all:
                if v.slot:
                    held[v.slot] = [x for x in held.get(v.slot, []) if x is not v]
                v.slot, v.method = None, ""
            for k in [k for k, x in held.items() if not x]:
                del held[k]
            rs = sorted([s for s in slots if s.row == row], key=lambda s: s.t)
            for vid, sid in order_match(sorted(rv_all, key=lambda v: v.t), rs, R).items():
                claim(next(v for v in vehs if v.id == vid), sid, "4 row re-fit")

    # Stage 5: rear-end roof-line stack
    for st in stacks:
        row_slots = sorted([sl for sl in slots if sl.row == st.row], key=lambda sl: -sl.t)
        free = [sl for sl in row_slots if sl.id not in held]
        for k, (ry, robust) in enumerate(zip(st.roofs, st.robust)):
            if k >= len(free):
                break
            pv = Vehicle(f"{st.id}.{k + 1}", "car", (st.box[0], ry - 5, st.box[2], ry + 5), "stack", st.score)
            pv.foot = free[k].poly.copy()
            pv.center = free[k].poly.mean(0)
            pv.t = free[k].t
            pv.zone = st.row
            pv.robust = robust
            pv.prior = k < getattr(st, "n_prior", 0)
            R[pv.id] = {sl.id: (1.0 if sl.id == free[k].id else 0.0) for sl in slots}
            vehs.append(pv)
            claim(pv, free[k].id, "5 stack")

    for v in vehs:
        if not v.slot:
            v.method = "no slot (aisle / outside bays)" if sum(R[v.id].values()) < 0.2 else "unresolved"
    return R, held, byid


def hidden_share(slot: Slot, vehs: list[Vehicle], to_photo, shape) -> float:
    """Fraction of bay in native photo coordinates occluded by vehicle boxes."""
    if shape is None or to_photo is None:
        return 0.0
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.int32(to_photo(slot.poly))], 1)
    tot = int(m.sum())
    if tot == 0:
        return 0.0
    cov = np.zeros(shape[:2], np.uint8)
    for v in vehs:
        x1, y1, x2, y2 = (int(max(0, q)) for q in v.box)
        cov[y1:y2, x1:x2] = 1
    return float((m & cov).sum() / tot)


def photo_px(tv):
    """Inverse transform: metres -> native photo pixels."""
    Hi = np.linalg.inv(tv.Hf).astype(np.float32)

    def f(pts):
        p = np.asarray(pts, np.float32).reshape(-1, 1, 2) * PPM
        return cv2.perspectiveTransform(p, Hi)[:, 0, :]

    return f


def slot_status(
    slots: list[Slot],
    vehs: list[Vehicle],
    R: dict,
    held: dict,
    valid_poly: np.ndarray,
    to_photo=None,
    shape=None,
    confirmed_free=(),
    site_report: bool = False,
) -> dict[str, list[dict]]:
    """Compute contextual area overlap, free width, visibility, and status per bay."""
    rows: dict[str, list[dict]] = {}
    for s in slots:
        occ = held.get(s.id, [])
        vis = overlap(s.poly, valid_poly) / max(s.area, 1e-4)
        cover = min(1.0, sum(overlap(v.foot, s.poly) for v in vehs) / max(s.area, 1e-4))
        blocked = []
        for v in vehs:
            if v.slot == s.id:
                continue
            lo, hi = float(v.foot[:, 1].min()), float(v.foot[:, 1].max())
            if hi > s.t0 and lo < s.t1 and overlap(v.foot, s.poly) > 0.12 * s.area:
                blocked.append((max(lo, s.t0), min(hi, s.t1)))
        blocked.sort()
        free, cur = 0.0, s.t0
        for a, b in blocked:
            if a > cur:
                free += a - cur
            cur = max(cur, b)
        free += max(0.0, s.t1 - cur)

        info: dict = {
            "slot": s,
            "vehicle": occ[0] if occ else None,
            "visible": vis,
            "free_width": free,
            "cover": cover,
            "hidden": hidden_share(s, vehs, to_photo, shape) if to_photo is not None and shape is not None else 0.0,
        }
        if occ:
            r = R[occ[0].id]
            others = [k for k, x in r.items() if k != s.id and x >= 0.25]
            info["status"] = "OCCUPIED"
            info["note"] = f"{occ[0].kind}; {r[s.id] * 100:.0f}% of footprint in slot" + (
                f"; also overlaps {', '.join(others)}" if others else ""
            )
        elif vis < 0.5:
            info["status"] = "UNKNOWN"
            info["note"] = f"only {vis * 100:.0f}% of bay inside frame"
        elif s.id in confirmed_free:
            info["status"] = "FREE"
            info["note"] = f"empty bay ({free:.1f} m clear between tyre lines)"
        elif info["hidden"] >= 0.5 and (free < 1.8 or cover >= 0.15):
            info["status"] = "UNKNOWN"
            info["note"] = "mostly hidden behind foreground vehicles"
        elif cover >= 0.5:
            info["status"] = "FREE (blocked)"
            info["note"] = "no vehicle assigned, neighbor overhang blocks bay"
        else:
            fits = "fits a car" if free >= MIN_CAR_GAP_M else "too narrow for a car"
            info["status"] = "FREE"
            info["note"] = f"{free:.1f} m free between neighbours ({fits})"
        rows.setdefault(s.row, []).append(info)
    return rows


def occupancy_summary(rows: dict[str, list[dict]]) -> tuple[int, int, int, float]:
    occ = sum(i["status"] == "OCCUPIED" for r in rows.values() for i in r)
    free = sum(i["status"] in ("FREE", "FREE (blocked)") for r in rows.values() for i in r)
    unk = sum(i["status"] == "UNKNOWN" for r in rows.values() for i in r)
    rate = occ / (occ + free) * 100.0 if occ + free else 0.0
    return occ, free, unk, rate
