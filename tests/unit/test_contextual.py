
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
