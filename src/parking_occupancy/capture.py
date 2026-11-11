"""
capture.py — Camera / video / still / folder frame acquisition.

Supports:
  - Live camera stream (OpenCV VideoCapture index or RTSP URL)
  - Single still image (--still)
  - Folder of images (--folder, evaluation mode)
  - Video file (--video)

All sources are exposed through a uniform Frame generator so the
rest of the pipeline never touches OpenCV VideoCapture directly.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


@dataclass(frozen=True)
class Frame:
    """A single acquired camera frame."""

    image: np.ndarray  # BGR, HxWx3
    timestamp: float   # monotonic seconds
    camera_id: str
    frame_index: int

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)


@dataclass
class CaptureConfig:
    """Parsed camera configuration from YAML."""

    camera_id: str
    source: str | int           # RTSP URL, file path, or webcam index
    reference_resolution: tuple[int, int]  # (w, h) used for polygon calibration
    polygon_config: Path
    device_id: str
    model_version: str
    model_path: str | None
    smoothing_consecutive_frames: int = 3
    freshness_threshold_s: float = 30.0
    heartbeat_interval_s: float = 60.0
    lot_code: str = "A"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "CaptureConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        source_raw = data["source"]
        # Numeric string → webcam index
        try:
            source: str | int = int(source_raw)
        except (ValueError, TypeError):
            source = source_raw or 0

        return cls(
            camera_id=data["camera_id"],
            source=source,
            reference_resolution=tuple(data["reference_resolution"]),  # type: ignore[arg-type]
            polygon_config=Path(data["polygon_config"]),
            device_id=data.get("device_id", "occupancy-dev-01"),
            model_version=data.get("model_version", "occupancy-demo-1"),
            model_path=data.get("model_path"),
            smoothing_consecutive_frames=int(data.get("smoothing_consecutive_frames", 3)),
            freshness_threshold_s=float(data.get("freshness_threshold_s", 30.0)),
            heartbeat_interval_s=float(data.get("heartbeat_interval_s", 60.0)),
            lot_code=data.get("lot_code", "A"),
            extra=data.get("extra", {}),
        )


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------

class CameraCapture:
    """
    Unified frame source that yields Frame objects.

    source can be:
      - Path to a directory → yields sorted image files
      - Path to an image file → yields that single frame
      - Path to a video file → yields video frames
      - int → webcam index
      - str URL → RTSP/HTTP camera stream
    """

    def __init__(
        self,
        source: str | int | Path,
        config: CaptureConfig,
        *,
        clock_fn=None,
    ) -> None:
        self.source = source
        self.config = config
        self._clock = clock_fn or time.monotonic

    # ------------------------------------------------------------------
    # Public generator
    # ------------------------------------------------------------------

    def frames(self) -> Iterator[Frame]:
        """Yield Frame objects from the configured source."""
        source = self.source
        if isinstance(source, Path) or (isinstance(source, str) and not source.isdigit()):
            p = Path(source)
            if p.is_dir():
                yield from self._folder_frames(p)
                return
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                yield from self._still_frames(p)
                return
            # Treat as video file
            yield from self._video_frames(p)
            return
        # Webcam index or RTSP URL
        yield from self._stream_frames(source)

    # ------------------------------------------------------------------
    # Private sources
    # ------------------------------------------------------------------

    def _still_frames(self, path: Path) -> Generator[Frame, None, None]:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image = self._resize_if_needed(image)
        yield Frame(
            image=image,
            timestamp=self._clock(),
            camera_id=self.config.camera_id,
            frame_index=0,
        )

    def _folder_frames(self, folder: Path) -> Generator[Frame, None, None]:
        images = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        if not images:
            raise ValueError(f"No supported images found in folder: {folder}")
        for idx, img_path in enumerate(images):
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue
            image = self._resize_if_needed(image)
            yield Frame(
                image=image,
                timestamp=self._clock(),
                camera_id=self.config.camera_id,
                frame_index=idx,
            )

    def _video_frames(self, path: Path) -> Generator[Frame, None, None]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self._resize_if_needed(frame)
                yield Frame(
                    image=frame,
                    timestamp=self._clock(),
                    camera_id=self.config.camera_id,
                    frame_index=idx,
                )
                idx += 1
        finally:
            cap.release()

    def _stream_frames(self, source: str | int) -> Generator[Frame, None, None]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera stream: {source!r}. "
                "Real camera streams require approved camera credentials."
            )
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed (source=%r). Frame dropped.", source)
                    break
                frame = self._resize_if_needed(frame)
                yield Frame(
                    image=frame,
                    timestamp=self._clock(),
                    camera_id=self.config.camera_id,
                    frame_index=idx,
                )
                idx += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize frame to reference_resolution if dimensions differ."""
        ref_w, ref_h = self.config.reference_resolution
        h, w = image.shape[:2]
        if (w, h) != (ref_w, ref_h):
            image = cv2.resize(image, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
        return image
