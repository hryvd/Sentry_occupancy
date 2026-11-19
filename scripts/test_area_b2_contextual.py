
from pathlib import Path

from shapely.affinity import scale

from parking_occupancy.geometry import PolygonMap
from parking_occupancy.contextual import assess_parking_lot


GEOMETRY = Path("configs/polygons/area_b2.json")

parking_map = PolygonMap.from_json(GEOMETRY)

print("SENTRY - AREA B2 CONTEXTUAL TEST")
print("Registered slots:", len(parking_map.spaces))


# TEST 1: Empty parking lot
print("\nTEST 1: EMPTY PARKING LOT")

empty_results = assess_parking_lot(
    GEOMETRY,
    vehicle_polygons=[]
)

for result in empty_results[:5]:
    print(result.slot_id, result.status)


# TEST 2: Simulate a parked vehicle
print("\nTEST 2: VEHICLE IN FIRST SLOT")

first_slot = parking_map.spaces[0]

# Create a synthetic vehicle covering
# 49% of the first parking slot's area.
vehicle = scale(
    first_slot._shapely,
    xfact=0.7,
    yfact=0.7,
    origin="centroid"
)

vehicle_points = list(
    vehicle.exterior.coords
)[:-1]

occupied_results = assess_parking_lot(
    GEOMETRY,
    vehicle_polygons=[vehicle_points]
)

first_result = next(
    r for r in occupied_results
    if r.slot_id == first_slot.space_id
)

print("Space ID:", first_result.slot_id)
print("Status:", first_result.status)
print("Slot area:", round(first_result.slot_area, 2))
print(
    "Consumed area:",
    round(first_result.consumed_area, 2)
)
print(
    "Occupied fraction:",
    round(first_result.occupied_fraction, 2)
)

assert len(empty_results) == len(parking_map.spaces)
assert all(r.status == "available" for r in empty_results)
assert first_result.status == "occupied"

print("\nAREA B2 INTEGRATION TEST PASSED")
