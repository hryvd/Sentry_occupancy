
from parking_occupancy.contextual import assess_occupancy


def test_occupied_slot():
    slot = [
        (0, 0), (100, 0),
        (100, 100), (0, 100)
    ]

    vehicle = [
        (10, 10), (90, 10),
        (90, 90), (10, 90)
    ]

    result = assess_occupancy(
        "A-C01", slot, [vehicle]
    )

    assert result.status == "occupied"
    assert result.occupied_fraction == 0.64


def test_available_slot():
    slot = [
        (0, 0), (100, 0),
        (100, 100), (0, 100)
    ]

    result = assess_occupancy(
        "A-C02", slot, []
    )

    assert result.status == "available"
    assert result.remaining_area == 10000



from parking_occupancy.contextual import assess_spillover


def test_cross_slot_spillover():
    slots = {
        "A": [(0, 0), (100, 0),
              (100, 100), (0, 100)],
        "B": [(100, 0), (200, 0),
              (200, 100), (100, 100)]
    }

    # Vehicle is primarily in A but extends into B.
    vehicle = [
        (20, 10), (128, 10),
        (128, 90), (20, 90)
    ]

    results = assess_spillover(
        slots,
        [vehicle]
    )

    assert results["A"]["contextual_label"] == \
        "occupied_direct"

    assert results["B"]["contextual_label"] == \
        "spillover_blocked"

    assert results["B"]["initial_recognition"] == \
        "occupied"


def test_two_adjacent_vehicles():
    slots = {
        "A": [(0, 0), (100, 0),
              (100, 100), (0, 100)],
        "B": [(100, 0), (200, 0),
              (200, 100), (100, 100)]
    }

    car_a = [
        (10, 10), (90, 10),
        (90, 90), (10, 90)
    ]

    car_b = [
        (110, 10), (190, 10),
        (190, 90), (110, 90)
    ]

    results = assess_spillover(
        slots,
        [car_a, car_b]
    )

    assert results["A"]["initial_recognition"] == \
        "occupied"

    assert results["B"]["initial_recognition"] == \
        "occupied"


def test_occluded_slot():
    slots = {
        "A": [(0, 0), (100, 0),
              (100, 100), (0, 100)]
    }

    results = assess_spillover(
        slots,
        [],
        occluded_slots=["A"]
    )

    assert results["A"]["contextual_label"] == \
        "occluded_unknown"

    assert results["A"]["initial_recognition"] == \
        "unknown"
