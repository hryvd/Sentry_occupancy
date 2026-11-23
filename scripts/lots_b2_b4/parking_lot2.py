#!/usr/bin/env python3
"""
parking_lot2.py
===============
Calibrated camera model and homography for Lot B4 (27 bays: L1-L8 cars,
R1-R10 angled cars, MB1-MB9 motorcycles) at reference resolution 2048x1536.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

AREA_B2_DIR = Path(__file__).resolve().parent.parent / "area_b2"
if str(AREA_B2_DIR) not in sys.path:
    sys.path.insert(0, str(AREA_B2_DIR))

import topview_slots as T  # noqa: E402

PPM = T.PPM
REF_WIDTH = 2048
REF_HEIGHT = 1536
SIZE_M = np.array([23.0, 30.0], dtype=np.float64)

# Photo ground polygon covering Lot B4 (clipped below horizon)
POLY = [(0, 120), (2048, 120), (2048, 1536), (0, 1536)]

# Reference calibrated detections (YOLOX-L + manual bays) from LOT B2 & B4
_REF_DET = [
    (660, 182, 1018, 377), (1017, 227, 1200, 453), (621, 372, 991, 568),
    (1038, 469, 1205, 709), (605, 526, 977, 702), (597, 667, 997, 862),
    (539, 837, 1015, 1053), (534, 1036, 1083, 1228), (494, 1332, 1124, 1528),
    (1231, 135, 1452, 248), (1286, 221, 1542, 329), (1323, 308, 1578, 410),
    (1341, 392, 1632, 497), (1365, 472, 1672, 615), (1412, 592, 1747, 724),
    (1422, 702, 1831, 878), (1506, 862, 1938, 1062), (1258, 864, 1733, 1517),
    (1584, 1055, 2050, 1278), (1677, 1306, 2046, 1485),
]
_REF_MAN = [
    (690, 290, 935, 425), (1040, 860, 1235, 1115),
    (301, 741, 512, 866), (340, 821, 549, 929), (287, 909, 519, 1010),
    (313, 1009, 500, 1078), (261, 1078, 478, 1145), (200, 1140, 437, 1254),
    (156, 1255, 415, 1354), (130, 1378, 383, 1457), (278, 1457, 357, 1496),
    (64, 855, 224, 1025),
]


def _fit_reference_homography() -> np.ndarray:
    """Fit 3x3 pixel->metres homography mapping 2048x1536 Lot B4 ground points to world metres."""
    allb = _REF_DET + _REF_MAN
    src, dst = [], []

    # Left kerb bays L1..L8
    for vid, k, width_m, len_m in [
        (1, 0, 1.95, 4.8), (21, 1, 1.5, 3.0), (3, 2, 1.85, 4.5), (5, 3, 1.85, 4.5),
        (6, 4, 1.85, 4.5), (7, 5, 1.85, 4.5), (8, 6, 1.85, 4.5), (9, 7, 1.85, 4.5),
    ]:
        x1, _, x2, y2 = allb[vid - 1]
        y0 = 9.0 + 2.4 * k
        cy = y0 + 1.2 + width_m / 2.0
        src += [[x1, y2], [x2, y2]]
        dst += [[6.8 - len_m / 2.0, cy], [6.8 + len_m / 2.0, cy]]

    # Right kerb angled bays R1..R10 (-10 deg)
    th = np.radians(-10.0)
    u = np.array([np.cos(th), np.sin(th)])
    n = np.array([u[1], -u[0]])
    if n[1] > 0:
        n = -n
    for vid, k in [(10, 0), (11, 1), (12, 2), (13, 3), (14, 4), (15, 5), (16, 6), (17, 7), (19, 8), (20, 9)]:
        x1, _, x2, y2 = allb[vid - 1]
        y0, y1b = 3.4 + 2.4 * k, 5.8 + 2.4 * k
        pt1 = np.array([14.0 + 0.045 * (y0 - 5.8), y0])
        pt2 = np.array([14.0 + 0.045 * (y1b - 5.8), y1b])
        c = np.mean([pt1, pt2, pt2 + u * 5.0, pt1 + u * 5.0], axis=0) - n * (1.85 / 2.0)
        src += [[x1, y2], [x2, y2]]
        dst += [(c - u * 2.25).tolist(), (c + u * 2.25).tolist()]

    # Lower-left motorcycle bays MB1..MB9
    for k in range(9):
        x1, _, x2, y2 = allb[23 + k - 1]
        y0 = 20.01 + 0.82 * k
        cy = y0 + 0.41 + 0.4
        src.append([(x1 + x2) / 2.0, y2])
        dst.append([3.05, cy])

    H_m, _ = cv2.findHomography(np.array(src, dtype=np.float64), np.array(dst, dtype=np.float64), cv2.RANSAC, 0.9)
    return H_m


_REF_H_M = _fit_reference_homography()


class TopViewB4:
    """Holds calibrated mapping: Lot B4 photo pixel <-> metres <-> top-view pixel."""

    def __init__(self, w: int = REF_WIDTH, h: int = REF_HEIGHT) -> None:
        self.w = int(w)
        self.h = int(h)
        sx = float(REF_WIDTH) / float(w)
        sy = float(REF_HEIGHT) / float(h)
        scale = np.diag([sx, sy, 1.0])
        self.H_m = _REF_H_M @ scale
        self.H0 = self.H_m
        self.origin = np.array([0.0, 0.0], dtype=np.float64)
        self.size_m = SIZE_M.copy()
        self.out_w = int(self.size_m[0] * PPM)
        self.out_h = int(self.size_m[1] * PPM)
        self.Hf = np.diag([float(PPM), float(PPM), 1.0]) @ self.H_m

    def to_metres(self, pts) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.H_m)[:, 0, :]

    def to_topview_px(self, pts) -> np.ndarray:
        return self.to_metres(pts) * PPM


def fit_camera(w: int = REF_WIDTH, h: int = REF_HEIGHT) -> TopViewB4:
    """Return calibrated TopViewB4 transform for given image width/height."""
    return TopViewB4(w, h)


def _build_prior_slots():
    """Back-project the 27 calibrated Lot B4 bay centers to (2048, 1536) anchor points."""
    Hi = np.linalg.inv(_REF_H_M)
    centers_m = []
    meta = []

    for k in range(8):
        y0 = 9.0 + 2.4 * k
        centers_m.append([6.8, y0 + 1.2])
        meta.append((f"L{k + 1}", "left", "confirmed", f"Lot B4 left bay L{k + 1}"))

    th = np.radians(-10.0)
    vec = np.array([np.cos(th), np.sin(th)]) * 5.0
    for k in range(10):
        y0, y1 = 3.4 + 2.4 * k, 5.8 + 2.4 * k
        pt1 = np.array([14.0 + 0.045 * (y0 - 5.8), y0])
        pt2 = np.array([14.0 + 0.045 * (y1 - 5.8), y1])
        c = np.mean([pt1, pt2, pt2 + vec, pt1 + vec], axis=0)
        centers_m.append(c.tolist())
        meta.append((f"R{k + 1}", "right", "confirmed", f"Lot B4 right angled bay R{k + 1}"))

    for k in range(9):
        y0 = 20.01 + 0.82 * k
        centers_m.append([3.05, y0 + 0.41])
        meta.append((f"MB{k + 1}", "moto", "confirmed", f"Lot B4 motorcycle bay MB{k + 1}"))

    pts_m = np.asarray(centers_m, dtype=np.float32).reshape(-1, 1, 2)
    pts_px = cv2.perspectiveTransform(pts_m, Hi)[:, 0, :]
    slots = []
    for (sid, row, status, note), (ax, ay) in zip(meta, pts_px):
        slots.append((sid, row, status, (int(round(ax)), int(round(ay))), note))
    return slots


SLOTS = _build_prior_slots()
