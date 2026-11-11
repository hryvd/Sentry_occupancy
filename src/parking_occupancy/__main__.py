"""
Entry point: python -m parking_occupancy --config <camera_yaml> --backend-mode mock
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from parking_occupancy.capture import CameraCapture, CaptureConfig
from parking_occupancy.inference import YOLODetector
from parking_occupancy.geometry import PolygonMap
from parking_occupancy.smoothing import StateSmoother
from parking_occupancy.freshness import FreshnessTracker
from parking_occupancy.events import EventEmitter
from parking_occupancy.backend_client import BackendClient, BackendMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("parking_occupancy")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="parking_occupancy",
        description="SENTRY parking occupancy detection service (3-view warp pipeline)",
    )
    p.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to camera YAML config file (e.g. configs/cameras/example.yaml)",
    )
    p.add_argument(
        "--backend-mode",
        choices=["mock", "real"],
        default="mock",
        help="Backend adapter mode. Use 'mock' for development (default: mock)",
    )
    p.add_argument(
        "--still",
        type=Path,
        default=None,
        help="Run on a single still image instead of live camera stream",
    )
    p.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Run on all images in a folder (evaluation mode)",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Run on a video file",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after N frames (useful for CI smoke tests)",
    )
    return p


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info("Starting SENTRY parking occupancy service (backend-mode=%s)", args.backend_mode)

    # -- Load camera config ---------------------------------------------------
    capture_cfg = CaptureConfig.from_yaml(args.config)
    logger.info("Loaded camera config: %s", capture_cfg.camera_id)

    # -- Load polygon map (includes per-camera homography) --------------------
    polygon_map = PolygonMap.from_json(capture_cfg.polygon_config)
    logger.info(
        "Loaded polygon map: %d spaces, calibration_version=%s",
        len(polygon_map.spaces),
        polygon_map.calibration_version,
    )

    # -- Instantiate pipeline components (all injectable for tests) -----------
    capture_source: str | Path
    if args.still:
        capture_source = args.still
    elif args.folder:
        capture_source = args.folder
    elif args.video:
        capture_source = args.video
    else:
        capture_source = capture_cfg.source  # live camera URL/index

    capture = CameraCapture(source=capture_source, config=capture_cfg)
    detector = YOLODetector.from_config(capture_cfg)
    smoother = StateSmoother.from_config(capture_cfg)
    freshness = FreshnessTracker.from_config(capture_cfg)

    backend_mode = BackendMode.MOCK if args.backend_mode == "mock" else BackendMode.REAL
    client = BackendClient(mode=backend_mode, config=capture_cfg)
    emitter = EventEmitter(
        producer_id=capture_cfg.device_id,
        model_version=capture_cfg.model_version,
        calibration_version=polygon_map.calibration_version,
        client=client,
    )

    # -- Main loop ------------------------------------------------------------
    frames_processed = 0
    try:
        for frame in capture.frames():
            if args.max_frames and frames_processed >= args.max_frames:
                logger.info("Reached max-frames=%d, stopping.", args.max_frames)
                break

            # 1. Detect vehicles in original (non-warped) frame
            detections = detector.detect(frame.image)

            # 2. Project detections to BEV via homography; classify each space
            space_states = polygon_map.classify(frame, detections)

            # 3. Apply temporal smoothing (hysteresis)
            smoothed = smoother.update(frame.timestamp, space_states)

            # 4. Update freshness tracker; mark stale spaces as unknown
            final_states = freshness.apply(frame.timestamp, smoothed)

            # 5. Emit changed / periodic observations
            emitter.emit_changes(frame.timestamp, final_states)

            frames_processed += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        emitter.flush()
        client.close()

    logger.info("Done. Processed %d frames.", frames_processed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
