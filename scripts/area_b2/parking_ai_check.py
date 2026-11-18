#!/usr/bin/env python3
"""
parking_ai_check.py
===================
Second opinion on the slot map from topview_slots.py, using TWO pretrained
networks that look at the photo in different ways:

  Model A  YOLOX-L (object detector)   "is there a vehicle box over this slot?"
           also re-run on zoomed crops of the far end, where cars are tiny.
  Model B  MiDaS-small (monocular depth)  "does something stand up above the
           ground plane here?"  A plane is fitted to bare concrete in the
           (relative) depth map; anything that rises above it is a vehicle.

A slot is marked
  both   - detector box AND height above ground   (strong evidence)
  one    - only one of the two                    (weak / check by hand)
  none   - neither                                (probably empty)

Usage
-----
    python parking_ai_check.py --image parking.jpg --out out_dir
(the two ONNX models are downloaded automatically from GitHub into ./models)

Needs: opencv-python, numpy, onnxruntime.  Imports SLOTS from topview_slots.py.

Honest limits: MiDaS-small is a coarse depth model (256x256 input) and its depth is
only relative, so it is a sanity check, not a measurement.  It is strong on the
near half of the lot and weak at the far end, which is exactly where the cars
stack up behind each other.  Neither network was trained on this camera; to get
a real parking-occupancy model, fine-tune a detector on labelled frames from the
actual fixed camera and keep the one-time homography from topview_slots.py.
"""
import argparse
import csv
import os
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

from topview_slots import SLOTS, ZONE_COLOR, yolox_vehicles

MODELS = {
    "yolox_l.onnx": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx",
    "midas_small.onnx": "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
}

# Bare concrete patches (x0, y0, x1, y1 in photo pixels) used to fit the ground plane in depth space.
GROUND_NEAR = [(1000, 1000, 1620, 1230), (1150, 480, 1260, 640), (1190, 650, 1290, 880),
               (150, 1060, 520, 1110), (560, 1130, 1000, 1250)]
GROUND_FAR = [(1110, 520, 1270, 700), (1210, 700, 1290, 840), (1040, 445, 1130, 470)]
NEAR_BAND = (0, 420, 2048, 1300)          # whole lot, one pass
FAR_CROP = (560, 430, 1560, 850)          # zoomed second pass for the far half
FAR_ZOOMS = {"far-left": (640, 430, 1010, 640), "far-right": (1230, 430, 1570, 640)}


def get_model(name, folder="models"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(MODELS[name], path)
    return path


# ---------------------------- Model A: detector ----------------------------
def detector_boxes(img, yolox_path):
    """Tiled full-image pass (from topview_slots) plus zoomed passes on the far ends."""
    boxes = [list(b[:4]) + [float(b[4])] for b in yolox_vehicles(img, yolox_path, 0.3)]
    sess = ort.InferenceSession(yolox_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    gx, sx = [], []
    for s in (8, 16, 32):
        n = 640 // s
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        gx.append(np.stack((xv, yv), 2).reshape(-1, 2))
        sx.append(np.full((n * n, 1), s))
    G, S = np.concatenate(gx), np.concatenate(sx)
    for x0, y0, x1, y1 in FAR_ZOOMS.values():
        crop = img[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        r = min(640 / ch, 640 / cw)
        pad = np.full((640, 640, 3), 114, np.uint8)
        rs = cv2.resize(crop, (int(cw * r), int(ch * r)), interpolation=cv2.INTER_CUBIC)
        pad[:rs.shape[0], :rs.shape[1]] = rs
        out = sess.run(None, {name: pad.transpose(2, 0, 1)[None].astype(np.float32)})[0][0]
        xy, wh, sc = (out[:, :2] + G) * S, np.exp(out[:, 2:4]) * S, out[:, 4:5] * out[:, 5:]
        for c in (2, 5, 7):
            m = sc[:, c] > 0.25
            for (cx, cy), (bw, bh), s in zip(xy[m], wh[m], sc[m, c]):
                boxes.append([(cx - bw / 2) / r + x0, (cy - bh / 2) / r + y0, (cx + bw / 2) / r + x0, (cy + bh / 2) / r + y0, float(s)])
    return np.array(boxes)


def anchor_in_box(boxes, p, pad=12):
    return any(b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad for b in boxes)


# ---------------------------- Model B: depth -------------------------------
class Depth:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def __call__(self, img):
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        o = self.sess.run(None, {"0": x.transpose(2, 0, 1)[None]})[0][0]      # relative inverse depth
        return cv2.resize(o, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)


def ground_plane_residual(depth, img, region, patches):
    """Fit disparity = a*x + b*y + c on bare-concrete patches; return (residual map, sigma, plane fn)."""
    x0, y0, x1, y1 = region
    d = depth(img[y0:y1, x0:x1])
    d = d / np.percentile(d, 99)
    xs, ys, ds = [], [], []
    for px0, py0, px1, py1 in patches:
        for y in range(py0, py1, 5):
            for x in range(px0, px1, 5):
                xs.append(x); ys.append(y); ds.append(d[y - y0, x - x0])
    xs, ys, ds = map(np.array, (xs, ys, ds))
    M = np.c_[xs, ys, np.ones(len(xs))]
    coef = np.linalg.lstsq(M, ds, rcond=None)[0]
    r = ds - M @ coef
    ok = np.abs(r) < 2.5 * r.std()                                # drop outliers (shadows, stains)
    coef = np.linalg.lstsq(M[ok], ds[ok], rcond=None)[0]
    sigma = (ds - M @ coef)[ok].std()
    yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
    plane = coef[0] * (xx + x0) + coef[1] * (yy + y0) + coef[2]
    return d - plane, sigma, (x0, y0)


def height_z(res, sigma, origin, p, w=14):
    x, y = int(p[0]) - origin[0], int(p[1]) - origin[1]
    patch = res[max(0, y - w):y + w, max(0, x - w):x + w]
    return float(patch.mean() / sigma) if patch.size else 0.0


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--models", default="models")
    ap.add_argument("--z", type=float, default=4.0, help="height z-score needed to call a slot 'above ground'")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    img = cv2.imread(args.image)

    yolox = get_model("yolox_l.onnx", args.models)
    midas = get_model("midas_small.onnx", args.models)

    boxes = detector_boxes(img, yolox)
    depth = Depth(midas)
    near, s_near, o_near = ground_plane_residual(depth, img, NEAR_BAND, GROUND_NEAR)
    far, s_far, o_far = ground_plane_residual(depth, img, FAR_CROP, GROUND_FAR)

    rows, vis = [], img.copy()
    for sid, zone, status, anchor, note in SLOTS:
        if anchor is None:
            rows.append([sid, zone, status, "", "", "", "none", "not visible in photo"])
            continue
        in_far = FAR_CROP[0] <= anchor[0] < FAR_CROP[2] and FAR_CROP[1] <= anchor[1] < FAR_CROP[3] and anchor[1] < 760
        z = height_z(*( (far, s_far, o_far) if in_far else (near, s_near, o_near) ), anchor)
        det = anchor_in_box(boxes, anchor)
        up = z >= args.z
        verdict = "both" if det and up else "one" if det or up else "none"
        rows.append([sid, zone, status, int(det), round(z, 1), "far" if in_far else "near", verdict, note])
        col = {"both": (60, 170, 60), "one": (0, 170, 255), "none": (60, 60, 220)}[verdict]
        cv2.circle(vis, tuple(map(int, anchor)), 25, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(vis, tuple(map(int, anchor)), 22, col, -1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(sid, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(vis, sid, (int(anchor[0]) - tw // 2, int(anchor[1]) + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # height-above-ground heat map (near pass) blended under the badges
    heat = np.zeros(img.shape[:2], np.float32)
    heat[NEAR_BAND[1]:NEAR_BAND[3]] = np.clip(near / (6 * s_near), 0, 1)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    base = cv2.addWeighted(img, 0.55, colour, 0.45, 0)
    mask = (heat > 0.02)[..., None]
    vis = np.where(mask, cv2.addWeighted(vis, 0.8, colour, 0.2, 0), vis)
    cv2.imwrite(os.path.join(args.out, "parking_ai_heightmap.jpg"), base, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(args.out, "parking_ai_check.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    with open(os.path.join(args.out, "parking_ai_check.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "zone", "status", "detector_box", "height_z", "depth_pass", "agreement", "note"])
        w.writerows(rows)

    tally = {}
    for r in rows:
        tally[r[6]] = tally.get(r[6], 0) + 1
    print("agreement of the two models:", tally)
    print("slots with weak support:", [r[0] for r in rows if r[6] != "both"])


if __name__ == "__main__":
    main()
