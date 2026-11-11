#!/usr/bin/env python3
"""
evaluate.py — Offline occupancy evaluation script.

Evaluates detector + classifier on annotated still frames.
Produces per-state confusion matrix, precision/recall, and false-available rate.

Usage:
  python scripts/evaluate.py \\
      --fixtures tests/fixtures/ \\
      --config configs/cameras/example.yaml \\
      --polygon-config configs/polygons/example.json
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluate")


@dataclass
class EvalRecord:
    space_id: str
    true_status: str  # available | occupied | unknown
    pred_status: str
    confidence: Optional[float]


@dataclass
class ConfusionMatrix:
    labels: list[str] = field(default_factory=lambda: ["available", "occupied", "unknown"])
    matrix: dict = field(default_factory=dict)

    def __post_init__(self):
        for t in self.labels:
            self.matrix[t] = {p: 0 for p in self.labels}

    def add(self, true: str, pred: str) -> None:
        t = true if true in self.labels else "unknown"
        p = pred if pred in self.labels else "unknown"
        self.matrix[t][p] += 1

    def precision(self, label: str) -> float:
        tp = self.matrix[label][label]
        total_pred = sum(self.matrix[t][label] for t in self.labels)
        return tp / total_pred if total_pred > 0 else 0.0

    def recall(self, label: str) -> float:
        tp = self.matrix[label][label]
        total_true = sum(self.matrix[label].values())
        return tp / total_true if total_true > 0 else 0.0

    def false_available_rate(self) -> float:
        """Rate at which occupied spaces are predicted as available."""
        fp_avail = self.matrix["occupied"]["available"]
        total_occupied = sum(self.matrix["occupied"].values())
        return fp_avail / total_occupied if total_occupied > 0 else 0.0

    def print_report(self) -> None:
        print("\n=== Confusion Matrix ===")
        header = f"{'True\\Pred':<12}" + "".join(f"{l:<12}" for l in self.labels)
        print(header)
        for t in self.labels:
            row = f"{t:<12}" + "".join(f"{self.matrix[t][p]:<12}" for p in self.labels)
            print(row)

        print("\n=== Per-class Precision / Recall ===")
        for label in ["available", "occupied"]:
            p = self.precision(label)
            r = self.recall(label)
            print(f"  {label}: precision={p:.3f}  recall={r:.3f}")

        print(f"\nFalse-available rate (occupied→predicted available): {self.false_available_rate():.3f}")
        print("NOTE: These metrics are on SYNTHETIC scenes — do not claim campus-level accuracy.")


def load_fixture_annotations(fixtures_dir: Path) -> dict[str, dict]:
    """
    Load ground-truth annotations from fixtures/annotations.json.
    Format: { "image_filename": { "space_id": "occupied|available|unknown", ... } }
    """
    ann_path = fixtures_dir / "annotations.json"
    if not ann_path.exists():
        logger.warning("No annotations.json found in %s — skipping evaluation.", fixtures_dir)
        return {}
    with open(ann_path) as f:
        return json.load(f)


def run_evaluation(
    fixtures_dir: Path,
    camera_config_path: Path,
    polygon_config_path: Path,
) -> None:
    from parking_occupancy.capture import CaptureConfig
    from parking_occupancy.geometry import PolygonMap
    from parking_occupancy.inference import SyntheticDetector

    config = CaptureConfig.from_yaml(camera_config_path)
    polygon_map = PolygonMap.from_json(polygon_config_path)
    annotations = load_fixture_annotations(fixtures_dir)

    if not annotations:
        print("No annotations found. Run with synthetic ground-truth fixtures.")
        return

    cm = ConfusionMatrix()
    records: list[EvalRecord] = []

    for img_name, gt_spaces in annotations.items():
        img_path = fixtures_dir / img_name
        if not img_path.exists():
            logger.warning("Fixture image not found: %s", img_path)
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Could not read: %s", img_path)
            continue

        # Resize to reference resolution
        ref_w, ref_h = config.reference_resolution
        image = cv2.resize(image, (ref_w, ref_h))

        # Use synthetic detector (no YOLO weights needed for evaluation with fixtures)
        # In real evaluation, replace with YOLODetector.from_config(config)
        detector = SyntheticDetector([])

        from parking_occupancy.capture import Frame
        frame = Frame(image=image, timestamp=0.0, camera_id=config.camera_id, frame_index=0)
        detections = detector.detect(image)
        states = polygon_map.classify(frame, detections)

        for state in states:
            gt = gt_spaces.get(state.space_id)
            if gt is None:
                continue
            cm.add(gt, state.physical_status.value)
            records.append(EvalRecord(
                space_id=state.space_id,
                true_status=gt,
                pred_status=state.physical_status.value,
                confidence=state.confidence,
            ))

    logger.info("Evaluated %d space-frame pairs.", len(records))
    cm.print_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTRY occupancy evaluation")
    parser.add_argument("--fixtures", type=Path, default=Path("tests/fixtures"))
    parser.add_argument("--config", type=Path, default=Path("configs/cameras/example.yaml"))
    parser.add_argument(
        "--polygon-config", type=Path, default=Path("configs/polygons/example.json")
    )
    args = parser.parse_args()
    run_evaluation(args.fixtures, args.config, args.polygon_config)


if __name__ == "__main__":
    main()

