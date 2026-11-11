"""
inference.py — YOLO vehicle detector adapter.

Detects cars and motorcycles in original (non-warped) camera frames.
Detection centroids are later projected through the homography matrix
in geometry.py to test polygon containment in BEV space.

Design rules:
  - Absence of detections NEVER means a space is empty on its own.
    Smoothing and polygon projection determine occupancy.
  - Missing model weights → raises MissingWeightsError (not silently returns nothing).
  - All vehicle detection logic is injectable for testing without real YOLO weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np

from parking_occupancy.capture import CaptureConfig

logger = logging.getLogger(__name__)

# COCO class IDs used in YOLOv8/YOLOv5 models
COCO_CLASS_CAR = 2
COCO_CLASS_MOTORCYCLE = 3
COCO_CLASS_TRUCK = 7
COCO_CLASS_BUS = 5

# We detect cars AND motorcycles (and optionally trucks/buses as occupancy evidence)
VEHICLE_CLASSES = {COCO_CLASS_CAR, COCO_CLASS_MOTORCYCLE}


class VehicleClass(str, Enum):
    CAR = "car"
    MOTORCYCLE = "motorcycle"


COCO_TO_VEHICLE_CLASS: dict[int, VehicleClass] = {
    COCO_CLASS_CAR: VehicleClass.CAR,
    COCO_CLASS_MOTORCYCLE: VehicleClass.MOTORCYCLE,
}


class MissingWeightsError(RuntimeError):
    """Raised when model weights file is not found or not configured."""


@dataclass(frozen=True)
class Detection:
    """A single vehicle detection in original image coordinates."""

    vehicle_class: VehicleClass
    confidence: float
    # Bounding box in original image pixels
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        """Centroid x in original image coords."""
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        """Centroid y in original image coords."""
        return (self.y1 + self.y2) / 2.0

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


class YOLODetector:
    """
    YOLOv8-based vehicle detector adapter.

    In mock/test mode (no weights), inject a SyntheticDetector instead.
    """

    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
    ) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._model = self._load_model(model_path)

    @classmethod
    def from_config(cls, config: CaptureConfig) -> "YOLODetector":
        if not config.model_path:
            raise MissingWeightsError(
                "MODEL_PATH is not configured. "
                "Provide a YOLOv8 weights file or inject a SyntheticDetector for testing."
            )
        return cls(model_path=Path(config.model_path))

    def _load_model(self, path: Path):
        """Load YOLOv8 model from weights file. CPU inference only."""
        if not path.exists():
            raise MissingWeightsError(
                f"Model weights not found: {path}. "
                "Download YOLOv8n.pt from https://github.com/ultralytics/assets/releases "
                "and set MODEL_PATH in .env"
            )
        try:
            from ultralytics import YOLO  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("ultralytics is not installed. Run: pip install ultralytics") from exc

        logger.info("Loading YOLO model from %s (CPU mode)", path)
        model = YOLO(str(path))
        model.to("cpu")
        return model

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run detection on a BGR image; return vehicle detections only."""
        results = self._model.predict(
            image,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            verbose=False,
            device="cpu",
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in VEHICLE_CLASSES:
                    continue
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    vehicle_class=COCO_TO_VEHICLE_CLASS[cls_id],
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return detections


class SyntheticDetector:
    """
    Injectable test double that returns pre-defined detections.

    Use this in unit tests and CI — no YOLO weights required.

    Example:
        detector = SyntheticDetector([
            Detection(VehicleClass.CAR, 0.92, 100, 200, 180, 280),
        ])
        detections = detector.detect(frame)
    """

    def __init__(self, detections: Sequence[Detection] | None = None) -> None:
        self._detections: list[Detection] = list(detections or [])

    def detect(self, image: np.ndarray) -> list[Detection]:  # noqa: ARG002
        return list(self._detections)

    def set_detections(self, detections: Sequence[Detection]) -> None:
        """Update detections between frames in test scenarios."""
        self._detections = list(detections)

