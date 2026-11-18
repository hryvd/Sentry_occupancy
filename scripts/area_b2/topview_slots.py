#!/usr/bin/env python3
"""
topview_slots.py
================
Turns an oblique photo of a parking lot into a metric top-down view, maps
numbered parking slots (L = left row, R = right row, M = middle block), and
writes a capacity / occupancy report.

Usage
-----
    python topview_slots.py --image parking.jpg --out out_dir
    python topview_slots.py --image parking.jpg --out out_dir --yolox yolox_l.onnx   # optional vehicle count check
    python topview_slots.py --image parking.jpg --calibrate                          # re-fit the camera model

Needs: opencv-python, numpy   (onnxruntime only for --yolox)

How the warp works
------------------
The lot is treated as a flat ground plane.  The camera is modelled as a pinhole
with focal length F (pixels), pitched down by atan(A / F) and rolled by ROLL
degrees.  Rotating that camera so it looks straight down gives a pure-rotation
homography  H = R_down * K^-1.  Ground coordinates are then scaled to metres
using the known wheelbase (~2.55 m) of two reference cars.

F / A / ROLL were fitted (see calibrate()) so that
  * two side-on cars at different depths give the same wheelbase,
  * the centre column of cars comes out perpendicular to the side-on cars,
  * cars in the left and right rows come out evenly spaced.

Limits: a homography is exact only for things lying ON the ground.  Car bodies
are 3-D, so roofs/sides are smeared away from the camera, and the far end of the
lot (small, grazing-angle pixels) is the least reliable part of the map.
"""
import argparse
import csv
import os

import cv2
import numpy as np

# --------------------------------------------------------------------------
# 1. Camera model + scene constants  (pixel coordinates of the ORIGINAL photo)
# --------------------------------------------------------------------------
F = 1000.0      # focal length in px (principal point assumed at image centre)
A = 620.0       # F * tan(pitch)  ->  pitch ~ 32 deg, horizon row ~ 148
ROLL = -7.0     # degrees
PPM = 60        # output resolution: pixels per metre in the top view

# Ground region used for the warp (everything above the far kerb and the window
# ledge at the bottom is dropped because it is not ground).
POLY = [(0, 800), (450, 640), (880, 462), (1290, 462), (1560, 560), (2048, 800),
        (2048, 1450), (1400, 1340), (1000, 1265), (600, 1195), (300, 1130), (0, 1070)]

# Wheel-contact pairs (same side of one car, ~2.55 m wheelbase) used for scale
# and for aligning the output so side-on cars point left-right.
REF_SEG = {"silver": ((600, 1055), (879, 1085)), "vios": ((493, 677), (624, 689))}
WHEELBASE_M = 2.55

# Extra calibration constraints (used by calibrate() only)
CENTRE_COLUMN = [(1003, 685), (1042, 602), (1070, 550)]            # white / dark / far-blue car
RIGHT_ROW_WHEELS = [((1582, 1115), 0), ((1468, 915), 1), ((1361, 721), 3), ((1325, 628), 4)]
LEFT_ROW_WHEELS = [((182, 1025), 0), ((414, 855), 1), ((493, 677), 2)]

# --------------------------------------------------------------------------
# 2. Slot table.  anchor = a point on the vehicle in the original photo.
#    status: confirmed   = vehicle clearly seen in that slot
#            probable    = vehicle seen by eye, weak support from the AI checks (far end)
#            unconfirmed = slot assumed from the 11-per-side layout, no vehicle seen
#            informal    = middle block (parked in the aisle, no marked bays)
# --------------------------------------------------------------------------
SLOTS = [
    # left row, near -> far
    ("L1",  "left",  "confirmed", (170, 890),  "red Hyundai Kona"),
    ("L2",  "left",  "confirmed", (345, 765),  "black Honda City"),
    ("L3",  "left",  "confirmed", (571, 628),  "dark grey sedan"),
    ("L4",  "left",  "confirmed", (680, 597),  "red hatchback/sedan"),
    ("L5",  "left",  "confirmed", (705, 555),  "grey hatchback"),
    ("L6",  "left",  "confirmed", (755, 538),  "black sedan"),
    ("L7",  "left",  "confirmed", (801, 520),  "black sedan"),
    ("L8",  "left",  "confirmed", (834, 492),  "black sedan"),
    ("L9",  "left",  "confirmed", (872, 472),  "black sedan"),
    ("L10", "left",  "confirmed", (918, 476),  "red car (far end)"),
    ("L11", "left",  "unconfirmed", None,      "no 11th vehicle clearly visible"),
    # right row, near -> far
    ("R1",  "right", "confirmed", (1780, 1040), "yellow hatchback"),
    ("R2",  "right", "confirmed", (1650, 860),  "grey sedan"),
    ("R3",  "right", "confirmed", (1570, 730),  "orange hatchback"),
    ("R4",  "right", "confirmed", (1510, 660),  "dark red SUV"),
    ("R5",  "right", "confirmed", (1440, 595),  "black sedan"),
    ("R6",  "right", "confirmed", (1410, 545),  "silver van"),
    ("R7",  "right", "confirmed", (1320, 522),  "dark car behind van"),
    ("R8",  "right", "confirmed", (1330, 500),  "dark car"),
    ("R9",  "right", "probable", (1345, 483),  "dark car (stacked behind van, count uncertain)"),
    ("R10", "right", "probable", (1360, 468),  "dark car (far end, count uncertain)"),
    ("R11", "right", "unconfirmed", None,       "no 11th vehicle clearly visible"),
    # middle block: front row of 3 + silver car parked crosswise + column of 3
    ("M1",  "middle", "informal", (690, 770),  "black sedan"),
    ("M2",  "middle", "informal", (915, 775),  "red sedan"),
    ("M3",  "middle", "informal", (1105, 800), "blue sedan"),
    ("M4",  "middle", "informal", (770, 960),  "silver sedan, parked crosswise"),
    ("M5",  "middle", "informal", (995, 630),  "white hatchback"),
    ("M6",  "middle", "informal", (1040, 555), "dark small car"),
    ("M7",  "middle", "informal", (1068, 520), "blue car (far end of column)"),
]
ZONE_COLOR = {"left": (230, 130, 40), "right": (40, 120, 235), "middle": (60, 170, 60)}  # BGR


# --------------------------------------------------------------------------
# 3. Geometry
# --------------------------------------------------------------------------
def rotation_to_down(w, h, f, a, roll_deg, yaw_deg=0.0):
    """3x3 matrix mapping source pixels -> rays of a virtual camera that looks straight down."""
    th = np.arctan(a / f)
    ph = np.radians(roll_deg)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    down = np.array([0, np.cos(th), np.sin(th)])
    Rr = np.array([[np.cos(ph), -np.sin(ph), 0], [np.sin(ph), np.cos(ph), 0], [0, 0, 1]])
    zv = Rr @ down
    zv /= np.linalg.norm(zv)
    xv = np.array([1.0, 0, 0])
    xv -= xv.dot(zv) * zv
    xv /= np.linalg.norm(xv)
    yv = np.cross(zv, xv)
    ya = np.radians(yaw_deg)
    Ry = np.array([[np.cos(ya), -np.sin(ya), 0], [np.sin(ya), np.cos(ya), 0], [0, 0, 1]])
    return Ry @ np.vstack([xv, yv, zv]) @ np.linalg.inv(K)


def ground(H0, pts):
    P = np.hstack([np.array(pts, float), np.ones((len(pts), 1))])
    D = (H0 @ P.T).T
    return D[:, :2] / D[:, 2:3]


class TopView:
    """Holds the calibrated mapping  photo pixel  <->  metres  <->  top-view pixel."""

    def __init__(self, w, h, f=F, a=A, roll=ROLL):
        self.w, self.h = w, h
        H0 = rotation_to_down(w, h, f, a, roll)
        g = ground(H0, REF_SEG["silver"])
        v = g[1] - g[0]
        yaw = -np.degrees(np.arctan2(v[1], v[0]))            # side-on cars point left-right
        self.H0 = rotation_to_down(w, h, f, a, roll, yaw)
        lens = [np.linalg.norm(np.diff(ground(self.H0, s), axis=0)) for s in REF_SEG.values()]
        self.mpu = WHEELBASE_M / np.mean(lens)                # metres per ground unit
        gp = ground(self.H0, POLY) * self.mpu
        self.origin = gp.min(0)
        self.size_m = gp.max(0) - gp.min(0)
        self.out_w = int(self.size_m[0] * PPM) + 1
        self.out_h = int(self.size_m[1] * PPM) + 1
        self.Hf = (np.diag([PPM, PPM, 1.0])
                   @ np.array([[1, 0, -self.origin[0]], [0, 1, -self.origin[1]], [0, 0, 1]])
                   @ np.diag([self.mpu, self.mpu, 1.0]) @ self.H0)

    def to_metres(self, pts):
        return ground(self.H0, pts) * self.mpu - self.origin

    def to_topview_px(self, pts):
        return self.to_metres(pts) * PPM


def render_topview(img, tv):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.array(POLY, np.int32)], 255)
    src = cv2.bitwise_and(img, img, mask=mask)
    warped = cv2.warpPerspective(src, tv.Hf, (tv.out_w, tv.out_h), flags=cv2.INTER_CUBIC)
    m = cv2.warpPerspective(mask, tv.Hf, (tv.out_w, tv.out_h), flags=cv2.INTER_LINEAR)
    alpha = (cv2.GaussianBlur(m, (5, 5), 0) / 255.0)[..., None]
    bg = np.full_like(warped, (26, 26, 26))
    return (warped * alpha + bg * (1 - alpha)).astype(np.uint8)


# --------------------------------------------------------------------------
# 4. Drawing helpers
# --------------------------------------------------------------------------
def badge(img, xy, text, color, r, scale, dashed=False):
    x, y = int(xy[0]), int(xy[1])
    cv2.circle(img, (x, y), r + 3, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (x, y), r, color, -1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(img, text, (x - tw // 2, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def scale_bar(img, ppm, metres=5, margin=24):
    h, w = img.shape[:2]
    x0, y0 = margin, h - margin
    cv2.rectangle(img, (x0 - 8, y0 - 34), (x0 + int(metres * ppm) + 8, y0 + 12), (0, 0, 0), -1)
    cv2.line(img, (x0, y0), (x0 + int(metres * ppm), y0), (255, 255, 255), 3)
    for k in (0, metres):
        cv2.line(img, (x0 + int(k * ppm), y0 - 7), (x0 + int(k * ppm), y0 + 7), (255, 255, 255), 3)
    cv2.putText(img, f"{metres} m", (x0 + 4, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def summary(slots=SLOTS):
    zones = {}
    for sid, zone, status, anchor, note in slots:
        z = zones.setdefault(zone, {"slots": 0, "seen": 0})
        z["slots"] += 1
        if anchor is not None:
            z["seen"] += 1
    return zones


def draw_legend(img, x, y, zones):
    lines = [("Left row", "left"), ("Right row", "right"), ("Middle block", "middle")]
    cv2.rectangle(img, (x - 10, y - 26), (x + 330, y + 18 + 30 * (len(lines) + 1)), (255, 255, 255), -1)
    cv2.rectangle(img, (x - 10, y - 26), (x + 330, y + 18 + 30 * (len(lines) + 1)), (60, 60, 60), 2)
    cv2.putText(img, "Slots   seen / mapped", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    for i, (label, z) in enumerate(lines, 1):
        yy = y + 30 * i
        cv2.circle(img, (x + 8, yy - 5), 9, ZONE_COLOR[z], -1, cv2.LINE_AA)
        cv2.putText(img, f"{label}:  {zones[z]['seen']} / {zones[z]['slots']}", (x + 28, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    tot_s = sum(v["slots"] for v in zones.values())
    tot_v = sum(v["seen"] for v in zones.values())
    cv2.putText(img, f"Total:  {tot_v} / {tot_s}", (x + 28, y + 30 * (len(lines) + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)


def slot_map_on_photo(img, tv=None):
    out = img.copy()
    for sid, zone, status, anchor, note in SLOTS:
        if anchor is None:
            continue
        badge(out, anchor, sid, ZONE_COLOR[zone], 24, 0.62)
    draw_legend(out, 40, 60, summary())
    return out


def slot_map_on_topview(top, tv):
    out = top.copy()
    for sid, zone, status, anchor, note in SLOTS:
        if anchor is None:
            continue
        p = tv.to_topview_px([anchor])[0]
        badge(out, p, sid, ZONE_COLOR[zone], 19, 0.5)
    scale_bar(out, PPM)
    draw_legend(out, 14, 44, summary())
    return out


def slot_plan(path_scale=34):
    """Clean schematic: standard 2.4 m x 5.0 m bays.  Not to scale with the real lot."""
    S = path_scale
    BW, BD = 2.4, 5.0
    W_m, H_m = 23.0, 37.0
    img = np.full((int(H_m * S), int(W_m * S), 3), 255, np.uint8)
    top_m = 3.0                                    # space for the title
    colors = {"confirmed": None, "probable": None, "unconfirmed": (235, 235, 235), "informal": (205, 230, 205)}

    def rect(x, y, w, h, sid, zone, status):
        p1, p2 = (int(x * S), int(y * S)), (int((x + w) * S), int((y + h) * S))
        k = 0.65 if status == "confirmed" else 0.35
        fill = colors[status] if colors[status] is not None else tuple(int((1 - k) * 255 + k * c) for c in ZONE_COLOR[zone])
        cv2.rectangle(img, p1, p2, fill, -1)
        if status == "unconfirmed":
            for xx in range(p1[0], p2[0], 10):
                cv2.line(img, (xx, p1[1]), (min(xx + (p2[1] - p1[1]), p2[0]), min(p1[1] + (p2[0] - xx), p2[1])), (200, 200, 200), 1)
        cv2.rectangle(img, p1, p2, (50, 50, 50), 2)
        (tw, th), _ = cv2.getTextSize(sid, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, sid, ((p1[0] + p2[0]) // 2 - tw // 2, (p1[1] + p2[1]) // 2 + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)

    status = {s[0]: s[2] for s in SLOTS}
    y_bottom = 29.5
    for i in range(11):                                         # left + right rows, near (bottom) -> far (top)
        y = y_bottom - (i + 1) * BW
        rect(1.0, y, BD, BW, f"L{i + 1}", "left", status[f"L{i + 1}"])
        rect(W_m - 1.0 - BD, y, BD, BW, f"R{i + 1}", "right", status[f"R{i + 1}"])
    cx = W_m / 2
    # middle front block (3 bays side by side) + silver car crosswise below it + column of 3 above
    yb = y_bottom - 3.4
    for k, sid in enumerate(("M1", "M2", "M3")):
        rect(cx - 1.5 * BW + k * BW, yb - BD, BW, BD, sid, "middle", status[sid])
    rect(cx - BD / 2, yb + 0.4, BD, BW, "M4", "middle", status["M4"])
    for k, sid in enumerate(("M5", "M6", "M7")):
        rect(cx - BW / 2, yb - BD - (k + 1) * BD, BW, BD, sid, "middle", status[sid])
    cv2.putText(img, "SLOT PLAN (schematic, not to scale)", (int(1 * S), int(1.2 * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "bays 2.4 m x 5.0 m  |  top of plan = far end of lot", (int(1 * S), int(2.0 * S)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 70, 70), 1, cv2.LINE_AA)
    # key
    kx, ky = int(12.5 * S), int(32.0 * S)
    cv2.putText(img, "solid = vehicle seen", (kx, ky), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "light colour = probable (far end)", (kx, ky + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "hatched grey = not confirmed", (kx, ky + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "pale green = middle block (aisle)", (kx, ky + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    draw_legend(img, int(1.6 * S), int(32.0 * S), summary())
    return img


# --------------------------------------------------------------------------
# 5. Optional: independent vehicle count with YOLOX (ONNX), tiled for small far cars
# --------------------------------------------------------------------------
def yolox_vehicles(img, model_path, score=0.3):
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    H, W = img.shape[:2]
    gx, sx = [], []
    for s in (8, 16, 32):
        n = 640 // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        gx.append(np.stack((xv, yv), 2).reshape(-1, 2))
        sx.append(np.full((n * n, 1), s))
    G, S = np.concatenate(gx), np.concatenate(sx)

    def run(crop, ox, oy, clip):
        ch, cw = crop.shape[:2]
        r = min(640 / ch, 640 / cw)
        pad = np.full((640, 640, 3), 114, np.uint8)
        rs = cv2.resize(crop, (int(cw * r), int(ch * r)))
        pad[:rs.shape[0], :rs.shape[1]] = rs
        out = sess.run(None, {name: pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy, wh = (out[:, :2] + G) * S, np.exp(out[:, 2:4]) * S
        sc = out[:, 4:5] * out[:, 5:]
        res = []
        for c in (2, 3, 5, 7):                                  # car, motorcycle, bus, truck
            m = sc[:, c] > score
            for (cx, cy), (bw, bh), s in zip(xy[m], wh[m], sc[m, c]):
                b = [(cx - bw / 2) / r + ox, (cy - bh / 2) / r + oy, (cx + bw / 2) / r + ox, (cy + bh / 2) / r + oy]
                if clip and ((ox > 0 and b[0] < ox + 6) or (ox + cw < W and b[2] > ox + cw - 6) or
                             (oy > 0 and b[1] < oy + 6) or (oy + ch < H and b[3] > oy + ch - 6)):
                    continue                                    # box cut by a tile edge
                res.append(b + [float(s), c])
        return res

    dets = run(img, 0, 0, False)
    for x0, y0, x1, y1 in [(0, 300, 1100, 1236), (948, 300, 2048, 1236), (0, 500, 1100, 1536), (948, 500, 2048, 1536), (500, 400, 1550, 1300)]:
        dets += run(img[y0:y1, x0:x1], x0, y0, True)
    d = np.array(dets)
    boxes = [[float(a), float(b), float(c - a), float(e - b)] for a, b, c, e, _, _ in d]
    idx = np.array(cv2.dnn.NMSBoxes(boxes, [float(s) for s in d[:, 4]], score, 0.35)).flatten()
    return d[idx]


# --------------------------------------------------------------------------
# 6. Camera calibration (re-fit F, A, ROLL from the constraints above)
# --------------------------------------------------------------------------
def calibrate(w, h):
    def metrics(f, a, roll):
        H0 = rotation_to_down(w, h, f, a, roll)
        g = ground(H0, REF_SEG["silver"])
        yaw = -np.degrees(np.arctan2(*(g[1] - g[0])[::-1]))
        H0 = rotation_to_down(w, h, f, a, roll, yaw)
        l1, l2 = [np.linalg.norm(np.diff(ground(H0, REF_SEG[k]), axis=0)) for k in ("silver", "vios")]
        gc = ground(H0, CENTRE_COLUMN)
        ang = np.degrees(np.arctan2(*(gc[2] - gc[0])[::-1]))

        def lin(rows):
            P = ground(H0, [p for p, _ in rows])
            idx = np.array([i for _, i in rows], float)
            M = np.c_[np.ones(len(idx)), idx]
            coef = np.linalg.lstsq(M, P, rcond=None)[0]
            return np.sqrt(((P - M @ coef) ** 2).sum(1).mean()) / np.linalg.norm(coef[1])
        return l2 / l1, abs(abs(ang) - 90), lin(RIGHT_ROW_WHEELS), lin(LEFT_ROW_WHEELS)

    best = []
    for f in (650, 750, 850, 1000):
        for a in np.arange(300, 900, 20):
            for roll in np.arange(-8, 8.1, 1):
                r, e, rr, lr = metrics(f, a, roll)
                best.append((abs(np.log(r)) * 25 + e * 0.6 + rr * 40 + lr * 40, f, a, roll))
    best.sort()
    for cost, f, a, roll in best[:5]:
        print(f"cost {cost:6.2f}  F={f}  A={a:.0f}  roll={roll:+.0f}  pitch={np.degrees(np.arctan(a / f)):.1f} deg")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--yolox", help="optional path to yolox_l.onnx for an independent vehicle count")
    ap.add_argument("--calibrate", action="store_true", help="re-fit the camera model and exit")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"cannot read {args.image}")
    h, w = img.shape[:2]
    if args.calibrate:
        calibrate(w, h)
        return
    os.makedirs(args.out, exist_ok=True)

    tv = TopView(w, h)
    top = render_topview(img, tv)
    top_plain = top.copy()
    scale_bar(top_plain, PPM)
    cv2.imwrite(os.path.join(args.out, "parking_topview.jpg"), top_plain, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(args.out, "parking_slotmap_topview.jpg"), slot_map_on_topview(top, tv), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(args.out, "parking_slotmap_photo.jpg"), slot_map_on_photo(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(args.out, "parking_slotmap_plan.png"), slot_plan())

    with open(os.path.join(args.out, "parking_slots.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["slot", "zone", "status", "photo_x", "photo_y", "x_m", "y_m", "note"])
        for sid, zone, status, anchor, note in SLOTS:
            if anchor:
                xm, ym = tv.to_metres([anchor])[0]
                wr.writerow([sid, zone, status, anchor[0], anchor[1], f"{xm:.1f}", f"{ym:.1f}", note])
            else:
                wr.writerow([sid, zone, status, "", "", "", "", note])

    z = summary()
    print(f"lot area covered by the warp: {tv.size_m[0]:.1f} m x {tv.size_m[1]:.1f} m (far end least reliable)")
    for k, v in z.items():
        print(f"{k:7s} slots mapped: {v['slots']:2d}   vehicles seen: {v['seen']:2d}")
    print(f"total   slots mapped: {sum(v['slots'] for v in z.values())}   vehicles seen: {sum(v['seen'] for v in z.values())}")

    if args.yolox:
        d = yolox_vehicles(img, args.yolox)
        cars = int((d[:, 5] == 2).sum())
        print(f"YOLOX check: {cars} cars + {int((d[:, 5] != 2).sum())} other vehicles detected (far, stacked cars are often merged)")


if __name__ == "__main__":
    main()
