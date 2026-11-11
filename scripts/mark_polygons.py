#!/usr/bin/env python3
"""
mark_polygons.py — Interactive polygon annotation and homography calibration tool.

Usage:
  python scripts/mark_polygons.py --image <path_to_reference_frame> \\
      --camera-id cam-lot-a-01 --lot A \\
      --output configs/polygons/lot_a_view1.json

Workflow:
  1. Load reference camera frame (a still image taken at calibration time).
  2. Click 4+ ground-plane correspondence points on the ORIGINAL image.
  3. Enter their matching coordinates in the BEV (bird's-eye) canvas.
  4. Tool computes H = cv2.findHomography(src, dst, RANSAC).
  5. Shows warped BEV preview — confirm lane markings are parallel.
  6. Click parking-space polygon vertices in the BEV image.
  7. Export JSON config with embedded H matrix, src/dst points, and space polygons.

Controls (during polygon annotation in BEV):
  - Left-click: add vertex to current polygon
  - 'n': finish current polygon (prompts for space_id, lot, compatibility)
  - 'u': undo last vertex
  - 's': save and exit
  - 'q': quit without saving
  - 'r': restart polygon annotation from scratch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class _AnnotationState:
    def __init__(self) -> None:
        self.src_points: list[list[float]] = []       # original image points (for H)
        self.dst_points: list[list[float]] = []       # BEV canvas points (for H)
        self.H: Optional[np.ndarray] = None
        self.bev_image: Optional[np.ndarray] = None
        self.current_polygon: list[tuple[float, float]] = []
        self.finished_spaces: list[dict] = []


# ---------------------------------------------------------------------------
# Homography step
# ---------------------------------------------------------------------------

def collect_correspondence_points(
    image: np.ndarray,
    bev_canvas_size: tuple[int, int],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Interactive: user clicks 4+ points on the original image,
    then types corresponding BEV canvas coordinates.
    Returns (src_points, dst_points).
    """
    print("\n=== STEP 1: Select ground-plane correspondence points ===")
    print("Click 4 or more points on the ORIGINAL image that lie on the ground plane.")
    print("Choose recognizable features: lane marking corners, parking bay corners, etc.")
    print("Press ENTER when done selecting points (minimum 4).")
    print()

    src_pts: list[list[float]] = []
    display = image.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            src_pts.append([float(x), float(y)])
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display, str(len(src_pts)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("STEP 1 — Click ground-plane points", display)
            print(f"  Point {len(src_pts)}: ({x}, {y}) in original image")

    cv2.namedWindow("STEP 1 — Click ground-plane points", cv2.WINDOW_NORMAL)
    cv2.imshow("STEP 1 — Click ground-plane points", display)
    cv2.setMouseCallback("STEP 1 — Click ground-plane points", on_click)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == 13 or key == ord('q'):  # ENTER or q
            break

    cv2.destroyAllWindows()

    if len(src_pts) < 4:
        print(f"ERROR: Need at least 4 points, got {len(src_pts)}. Exiting.")
        sys.exit(1)

    # Collect matching BEV coordinates
    bev_w, bev_h = bev_canvas_size
    print(f"\n=== STEP 2: Enter matching BEV canvas coordinates ===")
    print(f"BEV canvas size: {bev_w} x {bev_h} pixels")
    print("For each original point, enter the corresponding (x, y) in the BEV canvas.")
    print("Tip: use known distances (e.g., 1px = 10cm) for accurate scale.\n")

    dst_pts: list[list[float]] = []
    for i, (sx, sy) in enumerate(src_pts):
        raw = input(f"  Point {i+1} original=({sx:.0f},{sy:.0f}) → BEV (x,y): ").strip()
        try:
            x, y = [float(v) for v in raw.replace(",", " ").split()]
            dst_pts.append([x, y])
        except ValueError:
            print(f"  Invalid input '{raw}'. Using (0,0) as placeholder — redo calibration.")
            dst_pts.append([0.0, 0.0])

    return src_pts, dst_pts


def compute_and_preview_homography(
    image: np.ndarray,
    src_pts: list[list[float]],
    dst_pts: list[list[float]],
    bev_canvas_size: tuple[int, int],
) -> np.ndarray:
    """Compute H and show warped BEV. Returns H matrix."""
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        print("ERROR: Could not compute homography. Check your correspondence points.")
        sys.exit(1)

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"\nHomography computed. RANSAC inliers: {inliers}/{len(src_pts)}")

    bev_w, bev_h = bev_canvas_size
    bev = cv2.warpPerspective(image, H, (bev_w, bev_h))

    cv2.namedWindow("BEV Preview — confirm lane markings are parallel", cv2.WINDOW_NORMAL)
    cv2.imshow("BEV Preview — confirm lane markings are parallel", bev)
    print("\nBEV preview shown. Press any key to continue to polygon annotation.")
    print("If the perspective looks wrong, restart with better correspondence points.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H


# ---------------------------------------------------------------------------
# Polygon annotation step
# ---------------------------------------------------------------------------

def annotate_polygons(
    bev_image: np.ndarray,
) -> list[dict]:
    """
    Interactive polygon annotation on the BEV image.
    Returns list of space dicts with polygon_bev.
    """
    print("\n=== STEP 3: Annotate parking spaces on BEV image ===")
    print("Left-click: add vertex | 'n': finish polygon | 'u': undo | 's': save | 'q': quit")
    print()

    spaces: list[dict] = []
    current_poly: list[tuple[int, int]] = []
    display = bev_image.copy()

    colors = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255)]

    def redraw():
        nonlocal display
        display = bev_image.copy()
        # Draw completed spaces
        for i, sp in enumerate(spaces):
            pts = np.array(sp["polygon_bev"], dtype=np.int32)
            color = colors[i % len(colors)]
            cv2.polylines(display, [pts], True, color, 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(display, sp.get("lot_code", "?"), (cx - 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Draw current polygon in progress
        if current_poly:
            pts = np.array(current_poly, dtype=np.int32)
            cv2.polylines(display, [pts], False, (255, 255, 255), 1)
            for pt in current_poly:
                cv2.circle(display, pt, 4, (255, 255, 255), -1)
        cv2.imshow("BEV — Annotate Parking Spaces", display)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_poly.append((x, y))
            print(f"  Vertex added: ({x}, {y}) — total vertices: {len(current_poly)}")
            redraw()

    cv2.namedWindow("BEV — Annotate Parking Spaces", cv2.WINDOW_NORMAL)
    cv2.imshow("BEV — Annotate Parking Spaces", display)
    cv2.setMouseCallback("BEV — Annotate Parking Spaces", on_click)

    while True:
        key = cv2.waitKey(100) & 0xFF

        if key == ord('n'):  # Finish current polygon
            if len(current_poly) < 3:
                print("Need at least 3 vertices for a polygon. Keep clicking.")
                continue
            space_id = input("  Space UUID (or press ENTER for synthetic): ").strip()
            if not space_id:
                import uuid as _uuid
                space_id = f"30000000-0000-4000-8000-{len(spaces)+1:012d}"
            lot_code = input("  Lot code (A/B/C/D): ").strip().upper() or "A"
            compat = input("  Compatibility (car_only / motorcycle_only): ").strip()
            if compat not in ("car_only", "motorcycle_only"):
                print("  Invalid compatibility — defaulting to car_only")
                compat = "car_only"
            spaces.append({
                "space_id": space_id,
                "lot_code": lot_code,
                "compatibility": compat,
                "polygon_bev": [[float(p[0]), float(p[1])] for p in current_poly],
            })
            print(f"  Space {space_id} saved ({compat}, {len(current_poly)} vertices).")
            current_poly.clear()
            redraw()

        elif key == ord('u'):  # Undo last vertex
            if current_poly:
                removed = current_poly.pop()
                print(f"  Vertex {removed} removed.")
                redraw()

        elif key == ord('r'):  # Restart polygons
            spaces.clear()
            current_poly.clear()
            print("  All polygons cleared. Starting fresh.")
            redraw()

        elif key == ord('s') or key == ord('q'):
            break

    cv2.destroyAllWindows()
    return spaces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive parking space polygon annotation with homography calibration"
    )
    parser.add_argument("--image", type=Path, required=True,
                        help="Reference still image of the parking lot (taken at calibration time)")
    parser.add_argument("--camera-id", required=True,
                        help="Unique camera identifier (e.g. cam-lot-a-01)")
    parser.add_argument("--lot", default="A",
                        help="Lot code (A/B/C/D)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSON polygon config path")
    parser.add_argument("--bev-width", type=int, default=800,
                        help="BEV canvas width in pixels (default: 800)")
    parser.add_argument("--bev-height", type=int, default=600,
                        help="BEV canvas height in pixels (default: 600)")
    parser.add_argument("--calibration-version", default="v1",
                        help="Calibration version label (default: v1)")
    args = parser.parse_args()

    # Load reference image
    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: Could not read image: {args.image}")
        sys.exit(1)

    h, w = image.shape[:2]
    bev_canvas = (args.bev_width, args.bev_height)
    print(f"Loaded image: {w}×{h} pixels")
    print(f"BEV canvas: {bev_canvas[0]}×{bev_canvas[1]} pixels")

    # Step 1 & 2: collect correspondence points
    src_pts, dst_pts = collect_correspondence_points(image, bev_canvas)

    # Step 3: compute H and preview BEV
    H = compute_and_preview_homography(image, src_pts, dst_pts, bev_canvas)
    bev_image = cv2.warpPerspective(image, H, bev_canvas)

    # Step 4: annotate polygons on BEV
    spaces = annotate_polygons(bev_image)
    print(f"\nAnnotated {len(spaces)} spaces.")

    # Build output config
    config = {
        "calibration_version": f"lot-{args.lot.lower()}-{args.camera_id}-{args.calibration_version}",
        "camera_id": args.camera_id,
        "lot_code": args.lot,
        "reference_resolution": [w, h],
        "bev_canvas_size": list(bev_canvas),
        "homography_matrix": H.tolist(),
        "src_points": src_pts,
        "dst_points": dst_pts,
        "spaces": spaces,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPolygon config saved to: {args.output}")
    print(f"  Spaces: {len(spaces)}")
    print(f"  Calibration version: {config['calibration_version']}")


if __name__ == "__main__":
    main()
