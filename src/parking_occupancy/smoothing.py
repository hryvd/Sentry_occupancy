"""
smoothing.py — Temporal hysteresis state machine for per-space occupancy.

Rules:
  - A space transitions to OCCUPIED only after N consecutive frames show OCCUPIED.
  - A space transitions to AVAILABLE only after M consecutive frames show AVAILABLE.
    (M >= N to prevent flicker — leaving is harder to confirm than arriving.)
  - If a frame is missing or the space is UNKNOWN, the consecutive counter resets
    toward UNKNOWN — NEVER toward AVAILABLE.
  - Smoothed state is per-space and per-camera.

This prevents flicker caused by:
  - Momentary mis-detections (a pedestrian briefly overlapping a space)
  - Detection jitter on partially occluded vehicles
  - Lighting artifacts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from parking_occupancy.capture import CaptureConfig
from parking_occupancy.geometry import PhysicalStatus, SpaceState

logger = logging.getLogger(__name__)


@dataclass
class _SpaceBuffer:
    """Per-space rolling state buffer."""
    space_id: str
    confirmed_status: PhysicalStatus = PhysicalStatus.UNKNOWN
    pending_status: PhysicalStatus = PhysicalStatus.UNKNOWN
    consecutive_count: int = 0


class StateSmoother:
    """
    Applies configurable hysteresis smoothing to raw per-frame space classifications.

    Parameters
    ----------
    n_to_occupied : int
        Consecutive OCCUPIED frames needed to confirm occupied transition.
    n_to_available : int
        Consecutive AVAILABLE frames needed to confirm available transition.
        Should be >= n_to_occupied to reduce false vacancies.
    """

    def __init__(
        self,
        n_to_occupied: int = 3,
        n_to_available: int = 5,
    ) -> None:
        if n_to_occupied < 1:
            raise ValueError("n_to_occupied must be >= 1")
        if n_to_available < 1:
            raise ValueError("n_to_available must be >= 1")
        self._n_occ = n_to_occupied
        self._n_avail = n_to_available
        self._buffers: dict[str, _SpaceBuffer] = {}

    @classmethod
    def from_config(cls, config: CaptureConfig) -> "StateSmoother":
        n = config.smoothing_consecutive_frames
        return cls(n_to_occupied=n, n_to_available=max(n, n + 2))

    def update(
        self,
        timestamp: float,  # noqa: ARG002  (kept for future time-based smoothing)
        raw_states: Sequence[SpaceState],
    ) -> list[SpaceState]:
        """
        Apply smoothing to a list of raw per-frame SpaceStates.
        Returns smoothed SpaceStates — these are what gets forwarded to freshness.py.
        """
        smoothed: list[SpaceState] = []

        for state in raw_states:
            buf = self._buffers.setdefault(
                state.space_id,
                _SpaceBuffer(space_id=state.space_id),
            )
            buf = self._advance(buf, state.physical_status)
            self._buffers[state.space_id] = buf

            smoothed.append(SpaceState(
                space_id=state.space_id,
                lot_code=state.lot_code,
                compatibility=state.compatibility,
                physical_status=buf.confirmed_status,
                confidence=state.confidence if buf.confirmed_status != PhysicalStatus.UNKNOWN else None,
            ))

        return smoothed

    def _advance(self, buf: _SpaceBuffer, raw: PhysicalStatus) -> _SpaceBuffer:
        """
        Advance the hysteresis state machine for one space and one frame.
        """
        if raw == PhysicalStatus.UNKNOWN:
            # Unknown evidence breaks any pending transition; stay confirmed but do not clear
            buf.pending_status = PhysicalStatus.UNKNOWN
            buf.consecutive_count = 0
            return buf

        if raw == buf.pending_status:
            buf.consecutive_count += 1
        else:
            buf.pending_status = raw
            buf.consecutive_count = 1

        threshold = (
            self._n_occ if raw == PhysicalStatus.OCCUPIED else self._n_avail
        )

        if buf.consecutive_count >= threshold:
            if buf.confirmed_status != raw:
                logger.debug(
                    "Space %s: %s → %s (after %d consecutive frames)",
                    buf.space_id,
                    buf.confirmed_status.value,
                    raw.value,
                    buf.consecutive_count,
                )
            buf.confirmed_status = raw
            buf.consecutive_count = 0  # Reset for next transition

        return buf

    def reset_space(self, space_id: str) -> None:
        """Reset smoothing state for a space (e.g. after camera reconnect)."""
        if space_id in self._buffers:
            del self._buffers[space_id]
