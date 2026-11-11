"""
tests/fixtures/generate_synthetic_frames.py
Generate synthetic parking scene images for unit/evaluation tests.
Produces blank canvases with colored rectangles representing vehicles.
Run from module root: python tests/fixtures/generate_synthetic_frames.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import cv2


OUTPUT_DIR = Path(__file__).parent
W, H = 640, 480
BG_COLOR = (180, 180, 180)  # concrete gray
CAR_COLOR = (50, 50, 200)
CAR_W, CAR_H = 90, 50


def blank_lot() -> np.ndarray:
    img = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
    return img


def draw_car(img: np.ndarray, cx: int, cy: int) -> None:
    x1, y1 = cx - CAR_W // 2, cy - CAR_H // 2
    x2, y2 = cx + CAR_W // 2, cy + CAR_H // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), CAR_COLOR, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)


def main() -> None:
    # Space 1 center ≈ (100, 100), Space 2 ≈ (240, 85), Space 3 ≈ (100, 205) in BEV
    # In original 640x480 image with identity H these map approximately to same coords

    # Empty
    empty = blank_lot()
    cv2.imwrite(str(OUTPUT_DIR / "example_empty.jpg"), empty)
    print("Generated: example_empty.jpg")

    # Full
    full = blank_lot()
    draw_car(full, 100, 100)
    draw_car(full, 240, 85)
    draw_car(full, 100, 205)
    cv2.imwrite(str(OUTPUT_DIR / "example_full.jpg"), full)
    print("Generated: example_full.jpg")

    # Partial (space 1 only)
    partial = blank_lot()
    draw_car(partial, 100, 100)
    cv2.imwrite(str(OUTPUT_DIR / "example_partial.jpg"), partial)
    print("Generated: example_partial.jpg")

    print("Done.")


if __name__ == "__main__":
    main()
