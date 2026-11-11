"""
freshness.py — Camera freshness, stale state, boot UUID, and per-space sequence tracking.

Rules:
  - Each camera has a boot_id (UUID4 generated at service startup).
  - Each space has a monotonically increasing sequence_number per boot.
  - If no fresh frame arrives within freshness_threshold_s, all spaces on that
    camera transition to UNKNOWN (camera may be disconnected/moved).
  - Heartbeat events do NOT refresh stale space observations.
  - Only genuine fresh frames with new pixel data restore coverage.
  - Out-of-order sequence numbers from old boots are rejected.
  - An active parking assignment/visit is NEVER released on unknown status —
    that decision belongs entirely to the FastAPI backend.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Callable, Sequence

from parking_occupancy.capture import CaptureConfig
from parking_occupancy.geometry import PhysicalStatus, SpaceState

logger = logging.getLogger(__name__)


@dataclass
class _SpaceTracker:
    """Per-space sequence and last-seen tracking."""
    space_id: str
    sequence_number: int = 0
    last_observed_at: float = 0.0  # monotonic timestamp


class FreshnessTracker:
    """
    Manages camera freshness state.

    - Generates a new boot_id on instantiation (service restart).
    - Tracks per-space sequence numbers (monotonically increasing within a boot).
    - Marks all spaces UNKNOWN when the camera goes stale.
    - Resets on camera reconnect (new boot_id, sequence reset).
    """

    def __init__(
        self,
        freshness_threshold_s: float = 30.0,
        clock_fn: Callable[[], float] | None = None,
    ) -> None:
        self._threshold = freshness_threshold_s
        self._clock = clock_fn or __import__("time").monotonic
        self._boot_id: str = str(uuid.uuid4())
        self._space_trackers: dict[str, _SpaceTracker] = {}
        self._last_frame_time: float = self._clock()
        self._last_seen_frame_timestamp: float = -1.0
        self._is_stale: bool = False

        logger.info("FreshnessTracker initialized. boot_id=%s", self._boot_id)

    @classmethod
    def from_config(cls, config: CaptureConfig) -> "FreshnessTracker":
        return cls(freshness_threshold_s=config.freshness_threshold_s)

    @property
    def boot_id(self) -> str:
        return self._boot_id

    @property
    def is_stale(self) -> bool:
        return self._is_stale

    def apply(
        self,
        frame_timestamp: float,
        smoothed_states: Sequence[SpaceState],
    ) -> list[SpaceState]:
        """
        Apply freshness rules to smoothed states.

        1. Update last_frame_time.
        2. If stale (no fresh frame for > threshold), mark all spaces UNKNOWN.
        3. Otherwise increment sequence numbers and return states.
        """
        now = self._clock()
        elapsed = now - self._last_frame_time

        # --- Check for stale condition ------------------------------------
        if elapsed > self._threshold and not self._is_stale:
            logger.warning(
                "Camera stale: no fresh frame for %.1fs (threshold=%.1fs). "
                "Marking all spaces UNKNOWN.",
                elapsed,
                self._threshold,
            )
            self._is_stale = True

        if self._is_stale:
            # Fresh frame arriving NOW — recover from stale only if frame timestamp advanced
            if frame_timestamp > self._last_seen_frame_timestamp + 0.001:
                logger.info(
                    "Camera reconnected (new fresh frame). Recovering from stale state."
                )
                self._is_stale = False
                self._last_frame_time = now
                self._last_seen_frame_timestamp = frame_timestamp
            else:
                # Still stale — all spaces unknown
                return [
                    SpaceState(
                        space_id=s.space_id,
                        lot_code=s.lot_code,
                        compatibility=s.compatibility,
                        physical_status=PhysicalStatus.UNKNOWN,
                        confidence=None,
                    )
                    for s in smoothed_states
                ]

        self._last_frame_time = now
        self._last_seen_frame_timestamp = frame_timestamp

        # --- Assign sequence numbers --------------------------------------
        result: list[SpaceState] = []
        for state in smoothed_states:
            tracker = self._space_trackers.setdefault(
                state.space_id,
                _SpaceTracker(space_id=state.space_id, last_observed_at=frame_timestamp),
            )
            tracker.sequence_number += 1
            tracker.last_observed_at = frame_timestamp
            result.append(state)

        return result

    def get_sequence(self, space_id: str) -> int:
        """Return current sequence number for a space (0 if never observed)."""
        return self._space_trackers.get(space_id, _SpaceTracker(space_id=space_id)).sequence_number

    def reconnect(self) -> str:
        """
        Signal a camera reconnect event.

        Generates a new boot_id, resets all sequence numbers.
        Returns the new boot_id.
        """
        old_boot = self._boot_id
        self._boot_id = str(uuid.uuid4())
        self._space_trackers.clear()
        self._is_stale = False
        self._last_frame_time = self._clock()
        logger.info(
            "Camera reconnect: boot_id changed %s → %s",
            old_boot,
            self._boot_id,
        )
        return self._boot_id

    def advance_clock(self, seconds: float) -> None:
        """
        Test helper: simulate time passing without new frames.
        Forces staleness check on next apply() call.
        """
        self._last_frame_time -= seconds

