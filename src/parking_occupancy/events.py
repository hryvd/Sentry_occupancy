"""
events.py — Occupancy observation and heartbeat event serialization.

Produces OCCUPANCY_OBSERVATION and DEVICE_HEARTBEAT events matching the
CANDIDATE-2026-10-30.2 provisional schema (schema_version: "1.0.0").

Wire contract (provisional — not yet approved):
  POST /api/v1/observations/occupancy
  {
    "schema_version": "1.0.0",
    "event_id": "<uuid4>",
    "event_type": "occupancy_observation",
    "occurred_at": "<ISO8601 UTC>",
    "producer_id": "<device_id>",
    "correlation_id": null,          ← ALWAYS null for occupancy
    "data": {
      "lot_code": "A",
      "space_id": "<uuid>",
      "physical_status": "occupied|available|unknown",
      "confidence": 0.94,            ← null when status is unknown
      "boot_id": "<uuid4>",
      "sequence_number": 17,
      "calibration_version": "lot-a-polygons-1",
      "model_version": "occupancy-demo-1"
    }
  }

Important:
  - correlation_id is ALWAYS null — occupancy is not tied to an access attempt.
  - Original event_id/body/occurred_at are preserved on retry (no new UUID on resend).
  - Heartbeat events use event_type: "device_heartbeat" and have no space data.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from parking_occupancy.geometry import PhysicalStatus, SpaceState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_occupancy_event(
    *,
    producer_id: str,
    space_state: SpaceState,
    boot_id: str,
    sequence_number: int,
    calibration_version: str,
    model_version: str,
    occurred_at: str | None = None,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build a provisional occupancy_observation event payload."""
    return {
        "schema_version": "1.0.0",
        "event_id": event_id or str(uuid.uuid4()),
        "event_type": "occupancy_observation",
        "occurred_at": occurred_at or _utc_now_iso(),
        "producer_id": producer_id,
        "correlation_id": None,  # Always null for occupancy
        "data": {
            "lot_code": space_state.lot_code,
            "space_id": space_state.space_id,
            "physical_status": space_state.physical_status.value,
            "confidence": space_state.confidence
                if space_state.physical_status != PhysicalStatus.UNKNOWN
                else None,
            "boot_id": boot_id,
            "sequence_number": sequence_number,
            "calibration_version": calibration_version,
            "model_version": model_version,
        },
    }


def build_heartbeat_event(
    *,
    producer_id: str,
    boot_id: str,
    lot_code: str,
) -> dict[str, Any]:
    """Build a provisional device_heartbeat event payload."""
    return {
        "schema_version": "1.0.0",
        "event_id": str(uuid.uuid4()),
        "event_type": "device_heartbeat",
        "occurred_at": _utc_now_iso(),
        "producer_id": producer_id,
        "correlation_id": None,
        "data": {
            "boot_id": boot_id,
            "lot_code": lot_code,
        },
    }


# ---------------------------------------------------------------------------
# Event emitter (tracks changes and periodic refreshes)
# ---------------------------------------------------------------------------

@dataclass
class _SpaceRecord:
    last_emitted_status: PhysicalStatus | None = None
    last_emitted_at: float = 0.0


class EventEmitter:
    """
    Tracks per-space state and emits events on:
      1. State changes (occupied ↔ available ↔ unknown)
      2. Periodic heartbeat / refresh observations

    Preserves original event_id + occurred_at on retry.
    """

    def __init__(
        self,
        *,
        producer_id: str,
        model_version: str,
        calibration_version: str,
        client,  # BackendClient (injected)
        heartbeat_interval_s: float = 60.0,
        refresh_interval_s: float = 300.0,
    ) -> None:
        self._producer_id = producer_id
        self._model_version = model_version
        self._calibration_version = calibration_version
        self._client = client
        self._heartbeat_interval = heartbeat_interval_s
        self._refresh_interval = refresh_interval_s
        self._records: dict[str, _SpaceRecord] = {}
        self._last_heartbeat_at: float = 0.0

    def emit_changes(
        self,
        timestamp: float,
        states: Sequence[SpaceState],
    ) -> None:
        """Emit events for changed states and periodic refreshes."""
        boot_id = self._client.boot_id if hasattr(self._client, "boot_id") else "unknown"

        for state in states:
            rec = self._records.setdefault(state.space_id, _SpaceRecord())
            changed = rec.last_emitted_status != state.physical_status
            due_refresh = (timestamp - rec.last_emitted_at) >= self._refresh_interval

            if changed or due_refresh:
                seq = self._client.next_sequence(state.space_id)
                payload = build_occupancy_event(
                    producer_id=self._producer_id,
                    space_state=state,
                    boot_id=boot_id,
                    sequence_number=seq,
                    calibration_version=self._calibration_version,
                    model_version=self._model_version,
                )
                self._client.post_observation(payload)
                rec.last_emitted_status = state.physical_status
                rec.last_emitted_at = timestamp

        # Periodic heartbeat
        if (timestamp - self._last_heartbeat_at) >= self._heartbeat_interval:
            lot_code = states[0].lot_code if states else "?"
            hb = build_heartbeat_event(
                producer_id=self._producer_id,
                boot_id=boot_id,
                lot_code=lot_code,
            )
            self._client.post_heartbeat(hb)
            self._last_heartbeat_at = timestamp

    def flush(self) -> None:
        """Flush any pending retries in the client queue."""
        self._client.flush()

