#!/usr/bin/env python3
"""
mark_polygons.py — Interactive polygon annotation and homography calibration tool.

Usage:
  python scripts/mark_polygons.py --image <path_to_reference_frame> \
      --camera-id cam-lot-a-01 --lot A --camera-angle-type cctv_oblique \
      --output configs/polygons/lot_a_view1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


def collect_correspondence_points(
    image: np.ndarray,
    bev_canvas_size: tuple[int, int],
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Interactive: user clicks 4+ points on the original image,
    then types corresponding BEV canvas coordinates.
    """
    print("\n=== STEP 1: Select ground-plane correspondence points ===")
    print("Click 4 or more points on the ORIGINAL image that lie on the ground plane.")
    print("Press ENTER when done selecting points (minimum 4).\n")

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
        if key == 13 or key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(src_pts) < 4:
        print(f"ERROR: Need at least 4 points, got {len(src_pts)}. Exiting.")
        sys.exit(1)

    bev_w, bev_h = bev_canvas_size
    print(f"\n=== STEP 2: Enter matching BEV canvas coordinates ({bev_w} x {bev_h} px) ===")
    dst_pts: list[list[float]] = []
    for i, (sx, sy) in enumerate(src_pts):
        raw = input(f"  Point {i+1} original=({sx:.0f},{sy:.0f}) -> BEV (x,y): ").strip()
        try:
            x, y = [float(v) for v in raw.replace(",", " ").split()]
            dst_pts.append([x, y])
        except ValueError:
            print(f"  Invalid input '{raw}'. Using (0,0) as placeholder.")
            dst_pts.append([0.0, 0.0])

    return src_pts, dst_pts


def compute_reprojection_rmse(
    H: np.ndarray,
    src_pts: list[list[float]],
    dst_pts: list[list[float]],
) -> float:
    """Compute root-mean-square reprojection error in BEV pixels."""
    src_arr = np.array([[pt] for pt in src_pts], dtype=np.float64)
    dst_arr = np.array(dst_pts, dtype=np.float64)
    proj = cv2.perspectiveTransform(src_arr, H).reshape(-1, 2)
    errors = np.linalg.norm(proj - dst_arr, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def compute_and_preview_homography(
    image: np.ndarray,
    src_pts: list[list[float]],
    dst_pts: list[list[float]],
    bev_canvas_size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    """Compute H, report RMSE, and show side-by-side Raw + BEV preview."""
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        print("ERROR: Could not compute homography. Check your correspondence points.")
        sys.exit(1)

    inliers = int(mask.sum()) if mask is not None else 0
    rmse = compute_reprojection_rmse(H.astype(np.float64), src_pts, dst_pts)
    print(f"\nHomography computed. RANSAC inliers: {inliers}/{len(src_pts)} | Reprojection RMSE: {rmse:.2f} BEV px")

    bev_w, bev_h = bev_canvas_size
    bev = cv2.warpPerspective(image, H, (bev_w, bev_h))

    # Build side-by-side comparison preview
    raw_resized = cv2.resize(image, (bev_w, bev_h))
    for i, (sx, sy) in enumerate(src_pts):
        rx = int(sx * bev_w / image.shape[1])
        ry = int(sy * bev_h / image.shape[0])
        cv2.circle(raw_resized, (rx, ry), 5, (0, 0, 255), -1)
    for i, (dx, dy) in enumerate(dst_pts):
        cv2.circle(bev, (int(dx), int(dy)), 5, (0, 255, 0), -1)

    side_by_side = np.hstack([raw_resized, bev])
    cv2.namedWindow("Side-by-Side Preview (Left: Raw | Right: BEV)", cv2.WINDOW_NORMAL)
    cv2.imshow("Side-by-Side Preview (Left: Raw | Right: BEV)", side_by_side)
    print("\nPreview shown. Press any key to continue to polygon annotation.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return H, rmse


def annotate_polygons(
    bev_image: np.ndarray,
    default_lot: str = "A",
) -> list[dict]:
    """Interactive polygon annotation on the BEV image exporting slot_number and slot_area_px."""
    print("\n=== STEP 3: Annotate parking spaces on BEV image ===")
    print("Left-click: add vertex | 'n': finish polygon | 'u': undo | 'r': restart | 's': save\n")

    spaces: list[dict] = []
    current_poly: list[tuple[int, int]] = []
    display = bev_image.copy()
    colors = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255)]

    def redraw():
        nonlocal display
        display = bev_image.copy()
        for i, sp in enumerate(spaces):
            pts = np.array(sp["polygon_bev"], dtype=np.int32)
            color = colors[i % len(colors)]
            cv2.polylines(display, [pts], True, color, 2)
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(display, sp.get("slot_number", "?"), (cx - 18, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
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
        if key == ord('n'):
            if len(current_poly) < 3:
                print("Need at least 3 vertices for a polygon.")
                continue
            poly_obj = ShapelyPolygon(current_poly)
            if not poly_obj.is_valid or poly_obj.area < 50.0:
                print(f"Invalid or tiny polygon (area={poly_obj.area:.1f} px²). Redraw.")
                current_poly.clear()
                redraw()
                continue

            lot_code = input(f"  Lot code (default {default_lot}): ").strip().upper() or default_lot
            compat = input("  Compatibility (car_only / motorcycle_only): ").strip()
            if compat not in ("car_only", "motorcycle_only"):
                print("  Invalid compatibility — defaulting to car_only")
                compat = "car_only"
            prefix = "C" if compat == "car_only" else "M"
            default_slot_num = f"{lot_code}-{prefix}{len(spaces)+1:02d}"
            slot_number = input(f"  Slot number (default {default_slot_num}): ").strip() or default_slot_num
            space_id = input("  Space UUID (ENTER for synthetic): ").strip()
            if not space_id:
                space_id = f"30000000-0000-4000-8000-{len(spaces)+1:012d}"

            area_px = round(float(poly_obj.area), 2)
            spaces.append({
                "space_id": space_id,
                "slot_number": slot_number,
                "lot_code": lot_code,
                "compatibility": compat,
                "slot_area_px": area_px,
                "polygon_bev": [[float(p[0]), float(p[1])] for p in current_poly],
            })
            print(f"  Saved {slot_number} ({compat}, area={area_px} px²).")
            current_poly.clear()
            redraw()

        elif key == ord('u') and current_poly:
            current_poly.pop()
            redraw()
        elif key == ord('r'):
            spaces.clear()
            current_poly.clear()
            redraw()
        elif key in (ord('s'), ord('q')):
            break

    cv2.destroyAllWindows()
    return spaces


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive parking space polygon annotation with homography calibration"
    )
    parser.add_argument("--image", type=Path, required=True, help="Reference still image")
    parser.add_argument("--camera-id", required=True, help="Camera ID (e.g. cam-lot-a-01)")
    parser.add_argument("--lot", default="A", help="Lot code (A/B/C/D)")
    parser.add_argument(
        "--camera-angle-type",
        choices=["cctv_oblique", "optimized_angle"],
        default="cctv_oblique",
        help="Camera angle profile (cctv_oblique or optimized_angle)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output JSON polygon config path")
    parser.add_argument("--bev-width", type=int, default=800, help="BEV canvas width")
    parser.add_argument("--bev-height", type=int, default=600, help="BEV canvas height")
    parser.add_argument("--calibration-version", default="v1", help="Calibration version label")
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        print(f"ERROR: Could not read image: {args.image}")
        sys.exit(1)

    h, w = image.shape[:2]
    bev_canvas = (args.bev_width, args.bev_height)
    src_pts, dst_pts = collect_correspondence_points(image, bev_canvas)
    H, rmse = compute_and_preview_homography(image, src_pts, dst_pts, bev_canvas)
    bev_image = cv2.warpPerspective(image, H, bev_canvas)
    spaces = annotate_polygons(bev_image, default_lot=args.lot.upper())

    ground_anchors = [
        {"id": f"anchor_{i+1}", "image_xy": src_pts[i], "bev_xy": dst_pts[i]}
        for i in range(len(src_pts))
    ]

    config = {
        "calibration_version": f"lot-{args.lot.lower()}-{args.camera_id}-{args.calibration_version}",
        "camera_id": args.camera_id,
        "lot_code": args.lot.upper(),
        "camera_angle_type": args.camera_angle_type,
        "reference_resolution": [w, h],
        "bev_canvas_size": list(bev_canvas),
        "reprojection_rmse_px": round(rmse, 3),
        "homography_matrix": H.tolist(),
        "src_points": src_pts,
        "dst_points": dst_pts,
        "ground_anchors": ground_anchors,
        "spaces": spaces,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nPolygon config saved to: {args.output} ({len(spaces)} spaces, RMSE={rmse:.2f} px)")


if __name__ == "__main__":
    main()