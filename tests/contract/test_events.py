"""
tests/contract/test_events.py — Contract tests for events.py

Verifies that occupancy_observation and device_heartbeat payloads
conform to CANDIDATE-2026-10-30.2 provisional schema:
  - All required fields present
  - correlation_id is ALWAYS null
  - confidence is null when status is unknown
  - schema_version is "1.0.0"
  - Retry preserves original event_id and occurred_at
"""

from __future__ import annotations

import re
import uuid

import pytest

from parking_occupancy.events import build_heartbeat_event, build_occupancy_event
from parking_occupancy.geometry import PhysicalStatus, SpaceCompatibility, SpaceState


UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
ISO_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*Z$")


def occupied_state() -> SpaceState:
    return SpaceState(
        space_id="30000000-0000-4000-8000-000000000001",
        lot_code="A",
        compatibility=SpaceCompatibility.CAR_ONLY,
        physical_status=PhysicalStatus.OCCUPIED,
        confidence=0.94,
    )


def unknown_state() -> SpaceState:
    return SpaceState(
        space_id="30000000-0000-4000-8000-000000000001",
        lot_code="A",
        compatibility=SpaceCompatibility.CAR_ONLY,
        physical_status=PhysicalStatus.UNKNOWN,
        confidence=None,
    )


# ---------------------------------------------------------------------------
# Occupancy observation
# ---------------------------------------------------------------------------

def test_occupancy_event_required_fields():
    evt = build_occupancy_event(
        producer_id="occupancy-lot-a-01",
        space_state=occupied_state(),
        boot_id=str(uuid.uuid4()),
        sequence_number=1,
        calibration_version="lot-a-polygons-1",
        model_version="occupancy-demo-1",
    )
    assert evt["schema_version"] == "1.0.0"
    assert evt["event_type"] == "occupancy_observation"
    assert evt["correlation_id"] is None   # ALWAYS null
    assert UUID_RE.match(evt["event_id"])
    assert ISO_UTC_RE.match(evt["occurred_at"])

    data = evt["data"]
    assert data["lot_code"] == "A"
    assert data["space_id"] == "30000000-0000-4000-8000-000000000001"
    assert data["physical_status"] == "occupied"
    assert data["confidence"] == pytest.approx(0.94)
    assert data["sequence_number"] == 1
    assert data["calibration_version"] == "lot-a-polygons-1"
    assert data["model_version"] == "occupancy-demo-1"


def test_correlation_id_always_null():
    """correlation_id MUST be null for occupancy events — never set to an access attempt ID."""
    for status in [PhysicalStatus.OCCUPIED, PhysicalStatus.AVAILABLE, PhysicalStatus.UNKNOWN]:
        state = SpaceState("sp-1", "A", SpaceCompatibility.CAR_ONLY, status,
                           None if status == PhysicalStatus.UNKNOWN else 0.8)
        evt = build_occupancy_event(
            producer_id="dev-01",
            space_state=state,
            boot_id=str(uuid.uuid4()),
            sequence_number=1,
            calibration_version="v1",
            model_version="v1",
        )
        assert evt["correlation_id"] is None, f"correlation_id must be null for status={status}"


def test_confidence_null_when_unknown():
    """confidence MUST be null when physical_status is unknown."""
    evt = build_occupancy_event(
        producer_id="dev-01",
        space_state=unknown_state(),
        boot_id=str(uuid.uuid4()),
        sequence_number=5,
        calibration_version="v1",
        model_version="v1",
    )
    assert evt["data"]["physical_status"] == "unknown"
    assert evt["data"]["confidence"] is None


def test_confidence_present_when_occupied():
    evt = build_occupancy_event(
        producer_id="dev-01",
        space_state=occupied_state(),
        boot_id=str(uuid.uuid4()),
        sequence_number=1,
        calibration_version="v1",
        model_version="v1",
    )
    assert evt["data"]["confidence"] is not None
    assert 0.0 <= evt["data"]["confidence"] <= 1.0


def test_retry_preserves_event_id_and_timestamp():
    """
    On retry, the original event_id and occurred_at MUST be preserved.
    A retry is NOT a new event — it must be idempotent.
    """
    original = build_occupancy_event(
        producer_id="dev-01",
        space_state=occupied_state(),
        boot_id=str(uuid.uuid4()),
        sequence_number=3,
        calibration_version="v1",
        model_version="v1",
    )
    # Simulate retry by passing same event_id and occurred_at
    retry = build_occupancy_event(
        producer_id="dev-01",
        space_state=occupied_state(),
        boot_id=original["data"]["boot_id"],
        sequence_number=3,
        calibration_version="v1",
        model_version="v1",
        event_id=original["event_id"],        # preserved
        occurred_at=original["occurred_at"],  # preserved
    )
    assert retry["event_id"] == original["event_id"]
    assert retry["occurred_at"] == original["occurred_at"]


def test_sequence_numbers_differ_per_emission():
    state = occupied_state()
    boot_id = str(uuid.uuid4())
    evt1 = build_occupancy_event(
        producer_id="dev-01", space_state=state, boot_id=boot_id,
        sequence_number=10, calibration_version="v1", model_version="v1",
    )
    evt2 = build_occupancy_event(
        producer_id="dev-01", space_state=state, boot_id=boot_id,
        sequence_number=11, calibration_version="v1", model_version="v1",
    )
    assert evt2["data"]["sequence_number"] == evt1["data"]["sequence_number"] + 1


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def test_heartbeat_required_fields():
    hb = build_heartbeat_event(
        producer_id="occupancy-lot-a-01",
        boot_id=str(uuid.uuid4()),
        lot_code="A",
    )
    assert hb["schema_version"] == "1.0.0"
    assert hb["event_type"] == "device_heartbeat"
    assert hb["correlation_id"] is None
    assert UUID_RE.match(hb["event_id"])
    assert ISO_UTC_RE.match(hb["occurred_at"])
    assert hb["data"]["lot_code"] == "A"
    assert UUID_RE.match(hb["data"]["boot_id"])
