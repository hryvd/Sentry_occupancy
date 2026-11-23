#!/usr/bin/env python3
"""
build_b3.py - single source of truth for Area B3 (provisional).

Generates, from the hand-read layout below:
  configs/polygons/area_b3.json        slot polygons (photo pixels) + informal-zone boxes
  outputs/lot_b3/lot_b3_occupancy.csv  baseline in the fusion-CSV schema (manual visual labels, no model scores)
  outputs/lot_b3/lot_b3_overlay.png    overlay of every polygon on the reference photo (use it to check/adjust)

    python scripts/lot_b3/build_b3.py            # run from the package root

All coordinates are ORIGINAL pixels of ui/samples/area_b3_raw.jpg (2576 x 1932).
The photo is a hand-held frame, NOT the fixed camera frame: re-draw polygons on the real camera frame
(edit ROWS / PWD / CENTER below, re-run) before trusting the occupancy output.
"""
import csv
import json
import os

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
IMG = os.path.join(ROOT, "ui", "samples", "area_b3_raw.jpg")
UUID_BASE = "30000000-0000-4000-8000-000000000{:03d}"      # B2 = 101.., B4 = 201.., B3 = 301..

# Vehicle boxes (x1, y1, x2, y2) read off the photo, ordered top of photo -> bottom (far -> near).
LEFT = [  # angled row along the covered walkway; cars nose to the lower right
    (1140, 690, 1370, 795), (1050, 715, 1315, 825), (1065, 760, 1255, 865), (905, 765, 1210, 920),
    (775, 790, 1120, 970), (700, 880, 985, 1025), (500, 910, 900, 1120), (345, 1010, 730, 1200),
    (60, 1070, 620, 1350)]
RIGHT = [  # row along the kerb, cars side-on
    (1980, 725, 2280, 795), (1965, 765, 2250, 830), (1950, 800, 2240, 865), (1915, 840, 2240, 940),
    (1900, 895, 2245, 1020), (1855, 960, 2210, 1030), (1735, 1000, 2290, 1210), (1730, 1140, 2260, 1370),
    (1670, 1260, 2215, 1565)]
# Painted bay that is empty (quad in photo px)
PWD = [(1600, 1495), (2140, 1605), (2115, 1865), (1600, 1730)]            # R10, accessible bay, empty
CENTER = [(800, 1440), (930, 1235), (1270, 1130), (1300, 1300), (1130, 1485)]   # C1, painted bay, Camry
# Vehicles that are NOT in a bay (aisle / loading): tracked, never counted as bay occupancy
INFORMAL = [("A1", "jeepney", (1404, 773, 1610, 960)), ("A2", "sedan", (1240, 935, 1485, 1105)),
            ("A3", "sedan", (1539, 715, 1668, 837))]


def row_quads(boxes, axis_dy):
    """One parallelogram per vehicle: bay boundaries halfway between neighbouring vehicle centres,
    bay depth along the car axis (width of the box, tilt axis_dy)."""
    c = np.array([[(b[0] + b[2]) / 2, b[1] + 0.62 * (b[3] - b[1])] for b in boxes], float)
    n = len(c)
    mids = [(c[i] + c[i + 1]) / 2 for i in range(n - 1)]
    top = c[0] - (mids[0] - c[0])
    bot = c[-1] + (c[-1] - mids[-1])
    bounds = [top] + mids + [bot]
    quads = []
    for i, b in enumerate(boxes):
        w = b[2] - b[0]
        a = np.array([0.46 * w, axis_dy * 0.46 * w])
        p0, p1 = bounds[i], bounds[i + 1]
        quads.append([p0 - a, p0 + a, p1 + a, p1 - a])
    return quads


def build():
    slots = []
    lq = row_quads(LEFT, 0.25)
    rq = row_quads(RIGHT, -0.04)
    uid = 301
    spec = []
    for i, q in enumerate(lq, 1):
        spec.append(("L", f"L{i}", "left", q, "occupied", 1, "full", ""))
    for i, q in enumerate(rq, 1):
        spec.append(("R", f"R{i}", "right", q, "occupied", 1, "full", ""))
    spec.append(("R", "R10", "right", PWD, "available", 0, "full", "accessible (PWD) bay: reservation flag lives in backend metadata"))
    spec.append(("C", "C1", "center", CENTER, "occupied", 1, "full", "painted bay, low geometry confidence"))
    for _, sid, zone, q, st, fid, vis, note in spec:
        slots.append({
            "slot_id": sid, "zone": zone, "compatibility": "car_only",
            "space_id": UUID_BASE.format(uid),
            "polygon_px": [[round(float(x)), round(float(y))] for x, y in q],
            "visibility": vis, "baseline_state": st, "baseline_final_id": fid, "note": note})
        uid += 1
    return slots


def main():
    slots = build()
    cfg = {
        "area": "B3", "lot_code": "B", "camera_id": "cam-area-b3-01", "producer_id": "occupancy-area-b3-01",
        "calibration_version": "area-b3-4model-v0-provisional",
        "reference_image": "ui/samples/area_b3_raw.jpg", "image_size_px": [2576, 1932],
        "calibration_status": "PROVISIONAL - polygons read from a hand-held photo, not the fixed camera frame; no homography yet",
        "numbering": "L and R rows numbered top of photo -> bottom (far -> near); C1 = painted centre bay",
        "slot_count": len(slots), "slots": slots,
        "informal_zones": [{"id": i, "kind": k, "box_px": list(b), "counts_as_bay": False} for i, k, b in INFORMAL],
        "ignore": {"classes": ["person", "umbrella"],
                   "reason": "pedestrians cross the centre and right bays all day; they are never occupancy"},
    }
    with open(os.path.join(ROOT, "configs/polygons/area_b3.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(ROOT, "outputs/lot_b3/lot_b3_occupancy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow("area,slot_id,zone,compatibility,space_id,initial_state,model_1,model_2,model_3,model_4,mean,"
                   "final_state,final_id,backend_display_code".split(","))
        for s in slots:
            w.writerow(["B3", s["slot_id"], s["zone"], s["compatibility"], s["space_id"], s["baseline_state"],
                        "", "", "", "", "", s["baseline_state"],
                        "" if s["baseline_final_id"] is None else s["baseline_final_id"], "TBD (B-P...)"])
    # overlay
    im = Image.open(IMG).convert("RGB")
    ov = Image.new("RGBA", im.size)
    d = ImageDraw.Draw(ov)
    col = {"occupied": (230, 60, 60), "available": (60, 200, 80), "unknown": (150, 150, 150)}
    for s in slots:
        c = col[s["baseline_state"]]
        pts = [tuple(p) for p in s["polygon_px"]]
        d.polygon(pts, fill=c + (60,), outline=c + (255,))
        d.line(pts + [pts[0]], fill=c + (255,), width=4)
        cx, cy = np.mean(pts, 0)
        d.text((cx - 12, cy - 6), s["slot_id"], fill=(255, 255, 0, 255))
    for i, k, b in INFORMAL:
        d.rectangle(b, outline=(60, 140, 255, 255), width=4)
        d.text((b[0] + 6, b[1] + 6), i, fill=(120, 200, 255, 255))
    out = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
    out.save(os.path.join(ROOT, "outputs/lot_b3/lot_b3_overlay.png"))
    print(f"{len(slots)} slots written")


if __name__ == "__main__":
    main()
