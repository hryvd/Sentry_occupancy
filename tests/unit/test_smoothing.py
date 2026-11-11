"""
tests/unit/test_smoothing.py — Unit tests for smoothing.py

Tests:
  - occupied→available requires more evidence than available→occupied
  - Alternating detections do not flip state
  - Unknown evidence breaks pending transition
  - Missing data never settles to available
"""

from __future__ import annotations

import pytest

from parking_occupancy.geometry import PhysicalStatus, SpaceCompatibility, SpaceState
from parking_occupancy.smoothing import StateSmoother


def make_state(status: PhysicalStatus, space_id: str = "sp-1") -> SpaceState:
    return SpaceState(
        space_id=space_id,
        lot_code="A",
        compatibility=SpaceCompatibility.CAR_ONLY,
        physical_status=status,
        confidence=0.9 if status != PhysicalStatus.UNKNOWN else None,
    )


def feed(smoother: StateSmoother, statuses: list[PhysicalStatus]) -> list[PhysicalStatus]:
    results = []
    for s in statuses:
        out = smoother.update(0.0, [make_state(s)])
        results.append(out[0].physical_status)
    return results


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------

def test_occupied_requires_n_consecutive():
    smoother = StateSmoother(n_to_occupied=3, n_to_available=5)
    # 2 occupied frames → still unknown (not yet confirmed)
    out = feed(smoother, [PhysicalStatus.OCCUPIED, PhysicalStatus.OCCUPIED])
    assert out[-1] == PhysicalStatus.UNKNOWN
    # 3rd → confirmed
    out3 = smoother.update(0.0, [make_state(PhysicalStatus.OCCUPIED)])
    assert out3[0].physical_status == PhysicalStatus.OCCUPIED


def test_available_requires_more_frames_than_occupied():
    smoother = StateSmoother(n_to_occupied=2, n_to_available=4)
    # Confirm occupied first
    feed(smoother, [PhysicalStatus.OCCUPIED] * 2)
    # 3 available frames → still occupied
    out = feed(smoother, [PhysicalStatus.AVAILABLE] * 3)
    assert out[-1] == PhysicalStatus.OCCUPIED
    # 4th → available confirmed
    out4 = smoother.update(0.0, [make_state(PhysicalStatus.AVAILABLE)])
    assert out4[0].physical_status == PhysicalStatus.AVAILABLE


def test_alternating_detections_do_not_flip():
    """Jitter between occupied/available should not cause state changes."""
    smoother = StateSmoother(n_to_occupied=3, n_to_available=5)
    # Establish occupied
    feed(smoother, [PhysicalStatus.OCCUPIED] * 3)
    # Now alternate: should stay occupied because available never gets 5 consecutive
    alternating = [PhysicalStatus.AVAILABLE, PhysicalStatus.OCCUPIED] * 4
    out = feed(smoother, alternating)
    assert out[-1] == PhysicalStatus.OCCUPIED


def test_unknown_resets_pending():
    smoother = StateSmoother(n_to_occupied=3, n_to_available=5)
    # 2 occupied frames
    feed(smoother, [PhysicalStatus.OCCUPIED] * 2)
    # Unknown interrupts
    feed(smoother, [PhysicalStatus.UNKNOWN])
    # 1 more occupied → counter should have reset, so still unconfirmed
    out = smoother.update(0.0, [make_state(PhysicalStatus.OCCUPIED)])
    assert out[0].physical_status == PhysicalStatus.UNKNOWN


def test_missing_data_never_settles_to_available():
    """Frame drops (represented as UNKNOWN) must never produce AVAILABLE."""
    smoother = StateSmoother(n_to_occupied=2, n_to_available=3)
    # Establish occupied
    feed(smoother, [PhysicalStatus.OCCUPIED] * 2)
    # Only unknown evidence arrives
    out = feed(smoother, [PhysicalStatus.UNKNOWN] * 10)
    # Must NOT become available from unknown alone
    for status in out:
        assert status != PhysicalStatus.AVAILABLE


# ---------------------------------------------------------------------------
# Multi-space independence
# ---------------------------------------------------------------------------

def test_two_spaces_independent():
    smoother = StateSmoother(n_to_occupied=2, n_to_available=4)
    s1 = SpaceState("sp-1", "A", SpaceCompatibility.CAR_ONLY, PhysicalStatus.OCCUPIED, 0.9)
    s2 = SpaceState("sp-2", "A", SpaceCompatibility.CAR_ONLY, PhysicalStatus.AVAILABLE, 0.8)

    # Feed both spaces simultaneously
    for _ in range(2):
        out = smoother.update(0.0, [s1, s2])

    confirmed = {s.space_id: s.physical_status for s in out}
    assert confirmed["sp-1"] == PhysicalStatus.OCCUPIED
    assert confirmed["sp-2"] == PhysicalStatus.UNKNOWN  # Not yet 4 consecutive available


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_n_to_occupied():
    with pytest.raises(ValueError):
        StateSmoother(n_to_occupied=0)


def test_invalid_n_to_available():
    with pytest.raises(ValueError):
        StateSmoother(n_to_occupied=2, n_to_available=0)
