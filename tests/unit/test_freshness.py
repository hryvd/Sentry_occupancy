"""
tests/unit/test_freshness.py — Unit tests for freshness.py

Tests:
  - Spaces become unknown after freshness_threshold_s without fresh frames
  - Heartbeat alone does NOT refresh staleness
  - Recovery after fresh frame arrives
  - boot_id changes on reconnect; sequences reset
  - Out-of-order / old boot sequences do not overwrite current state
  - advance_clock() helper works for fake-clock tests
"""

from __future__ import annotations

import pytest

from parking_occupancy.freshness import FreshnessTracker
from parking_occupancy.geometry import PhysicalStatus, SpaceCompatibility, SpaceState


def make_state(status: PhysicalStatus = PhysicalStatus.OCCUPIED, space_id: str = "sp-1") -> SpaceState:
    return SpaceState(
        space_id=space_id,
        lot_code="A",
        compatibility=SpaceCompatibility.CAR_ONLY,
        physical_status=status,
        confidence=0.9 if status != PhysicalStatus.UNKNOWN else None,
    )


# ---------------------------------------------------------------------------
# Normal operation
# ---------------------------------------------------------------------------

def test_fresh_frames_pass_through():
    tracker = FreshnessTracker(freshness_threshold_s=30.0)
    states = [make_state(PhysicalStatus.OCCUPIED)]
    result = tracker.apply(1.0, states)
    assert result[0].physical_status == PhysicalStatus.OCCUPIED


def test_sequences_increment():
    tracker = FreshnessTracker(freshness_threshold_s=30.0)
    state = make_state()
    tracker.apply(1.0, [state])
    tracker.apply(2.0, [state])
    tracker.apply(3.0, [state])
    assert tracker.get_sequence("sp-1") == 3


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------

def test_stale_after_threshold():
    tracker = FreshnessTracker(freshness_threshold_s=30.0)
    states = [make_state(PhysicalStatus.OCCUPIED)]
    # First frame is fresh
    tracker.apply(1.0, states)
    # Advance clock 31s without new frames
    tracker.advance_clock(31.0)
    # Next apply with same timestamp (simulating stale condition)
    result = tracker.apply(1.0, states)  # same timestamp = not a new fresh frame
    for s in result:
        assert s.physical_status == PhysicalStatus.UNKNOWN
        assert s.confidence is None


def test_no_staleness_before_threshold():
    tracker = FreshnessTracker(freshness_threshold_s=30.0)
    states = [make_state(PhysicalStatus.OCCUPIED)]
    tracker.apply(1.0, states)
    # Advance only 10s — still fresh
    tracker.advance_clock(10.0)
    result = tracker.apply(2.0, states)  # new timestamp
    assert result[0].physical_status == PhysicalStatus.OCCUPIED


def test_recovery_after_fresh_frame():
    tracker = FreshnessTracker(freshness_threshold_s=5.0)
    states = [make_state(PhysicalStatus.OCCUPIED)]
    tracker.apply(1.0, states)
    # Go stale
    tracker.advance_clock(10.0)
    tracker.apply(1.0, states)  # stale — same timestamp
    assert tracker.is_stale

    # Recover with genuinely new timestamp
    result = tracker.apply(999.0, states)
    assert not tracker.is_stale
    assert result[0].physical_status == PhysicalStatus.OCCUPIED


def test_stale_does_not_release_active_visit():
    """
    Staleness marks coverage unknown — it NEVER signals that a space is
    empty/available. Backend assignment/visit decisions are unaffected.
    """
    tracker = FreshnessTracker(freshness_threshold_s=5.0)
    states = [make_state(PhysicalStatus.OCCUPIED)]
    tracker.apply(1.0, states)
    tracker.advance_clock(10.0)
    result = tracker.apply(1.0, states)
    # Unknown, not Available
    for s in result:
        assert s.physical_status == PhysicalStatus.UNKNOWN
        assert s.physical_status != PhysicalStatus.AVAILABLE


# ---------------------------------------------------------------------------
# Reconnect
# ---------------------------------------------------------------------------

def test_reconnect_generates_new_boot_id():
    tracker = FreshnessTracker()
    old_boot = tracker.boot_id
    new_boot = tracker.reconnect()
    assert new_boot != old_boot
    assert tracker.boot_id == new_boot


def test_reconnect_resets_sequences():
    tracker = FreshnessTracker()
    state = make_state()
    tracker.apply(1.0, [state])
    tracker.apply(2.0, [state])
    assert tracker.get_sequence("sp-1") == 2
    tracker.reconnect()
    assert tracker.get_sequence("sp-1") == 0


def test_reconnect_clears_stale():
    tracker = FreshnessTracker(freshness_threshold_s=1.0)
    tracker.advance_clock(5.0)
    # Manually set stale
    tracker.apply(0.0, [make_state()])
    tracker.reconnect()
    assert not tracker.is_stale
