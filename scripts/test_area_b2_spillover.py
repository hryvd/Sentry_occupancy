
import json
from pathlib import Path

from shapely.geometry import Polygon
from shapely.ops import unary_union

from parking_occupancy.contextual import assess_spillover


GEOMETRY_PATH = Path("configs/polygons/area_b2.json")

with GEOMETRY_PATH.open(encoding="utf-8") as file:
    geometry = json.load(file)

spaces = geometry["spaces"]

# Use actual configured BEV polygons.
slots = {
    space["space_id"]: space["polygon_bev"]
    for space in spaces
}

slot_names = {
    space["space_id"]: space["slot_number"]
    for space in spaces
}

print("=" * 55)
print("SENTRY - AREA B2 CONTEXTUAL AWARENESS")
print("=" * 55)
print("Registered parking slots:", len(slots))

assert len(slots) == 29


# SCENARIO 1: No simulated vehicles
print("\nSCENARIO 1: EMPTY PARKING LOT")

empty_results = assess_spillover(
    slots=slots,
    vehicles=[]
)

assert all(
    result["initial_recognition"] == "available"
    for result in empty_results.values()
)

print("All 29 slots classified as available.")


# SCENARIO 2: One simulated vehicle spanning two slots
print("\nSCENARIO 2: CROSS-SLOT SPILLOVER")

by_number = {
    space["slot_number"]: space
    for space in spaces
}

slot_a = by_number["L1"]
slot_b = by_number["L2"]

polygon_a = Polygon(slot_a["polygon_bev"])
polygon_b = Polygon(slot_b["polygon_bev"])

# Artificial worst-case footprint:
# one large polygon connecting and covering L1 and L2.
vehicle_polygon = unary_union([
    polygon_a,
    polygon_b
]).convex_hull

vehicle_points = list(
    vehicle_polygon.exterior.coords
)[:-1]

results = assess_spillover(
    slots=slots,
    vehicles=[vehicle_points]
)

for number in ["L1", "L2"]:
    space = by_number[number]
    result = results[space["space_id"]]

    print(f"\nParking Slot: {number}")
    print("Status:", result["initial_recognition"])
    print("Context:", result["contextual_label"])
    print("Consumed Area:", result["consumed_area"])
    print("Remaining Area:", result["remaining_area"])
    print("Occupied Fraction:", result["occupied_fraction"])

blocked_slots = [
    slot_names[slot_id]
    for slot_id, result in results.items()
    if result["contextual_label"] == "spillover_blocked"
]

print("\nSpillover-blocked slots:", blocked_slots)

assert blocked_slots, (
    "No spillover-blocked slot detected. "
    "Inspect the layout and simulated footprint."
)

print("\nAREA B2 SPILLOVER SIMULATION PASSED")
