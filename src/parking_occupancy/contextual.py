
from dataclasses import dataclass

from shapely.geometry import Polygon
from shapely.ops import unary_union


@dataclass
class ContextualAssessment:
    slot_id: str
    slot_area: float
    consumed_area: float
    remaining_area: float
    occupied_fraction: float
    status: str


def assess_occupancy(slot_id, slot_points, vehicles):
    slot = Polygon(slot_points)

    if not slot.is_valid or slot.area <= 0:
        raise ValueError("Invalid parking slot polygon")

    vehicle_polygons = [
        Polygon(points) for points in vehicles
    ]

    if any(not p.is_valid for p in vehicle_polygons):
        raise ValueError("Invalid vehicle polygon")

    if vehicle_polygons:
        combined = unary_union(vehicle_polygons)
        consumed = slot.intersection(combined).area
    else:
        consumed = 0.0

    area = slot.area
    remaining = area - consumed
    fraction = consumed / area

    if fraction >= 0.35:
        status = "occupied"
    elif fraction < 0.10:
        status = "available"
    else:
        status = "unknown"

    return ContextualAssessment(
        slot_id=slot_id,
        slot_area=area,
        consumed_area=consumed,
        remaining_area=remaining,
        occupied_fraction=fraction,
        status=status
    )



from pathlib import Path

from parking_occupancy.geometry import PolygonMap


def assess_parking_lot(
    geometry_path,
    vehicle_polygons
):
    """
    Assess every registered parking slot.

    geometry_path: Path to SENTRY polygon JSON.
    vehicle_polygons: List of vehicle polygons in BEV coordinates.
    """

    parking_map = PolygonMap.from_json(
        Path(geometry_path)
    )

    results = []

    for space in parking_map.spaces:
        assessment = assess_occupancy(
            slot_id=space.space_id,
            slot_points=space.polygon_bev,
            vehicles=vehicle_polygons
        )

        results.append(assessment)

    return results



from shapely.geometry import box


def _has_usable_space(slot, remaining, width_ratio=0.75,
                      height_ratio=0.70):
    """
    Approximate whether a vehicle-sized rectangle fits in
    the remaining free space.

    Assumes the slot is roughly axis-aligned in BEV.
    """
    minx, miny, maxx, maxy = slot.bounds

    slot_width = maxx - minx
    slot_height = maxy - miny

    required_width = slot_width * width_ratio
    required_height = slot_height * height_ratio

    if required_width <= 0 or required_height <= 0:
        return False

    # Sample possible rectangle positions.
    for i in range(21):
        for j in range(21):
            x = minx + (
                slot_width - required_width
            ) * i / 20

            y = miny + (
                slot_height - required_height
            ) * j / 20

            candidate = box(
                x,
                y,
                x + required_width,
                y + required_height
            )

            if remaining.covers(candidate):
                return True

    return False


def assess_spillover(slots, vehicles, occluded_slots=None):
    """
    Assess direct occupancy and spillover across parking slots.

    slots:
        Dictionary mapping space_id to BEV polygon coordinates.

    vehicles:
        List of vehicle footprint polygons in BEV coordinates.

    occluded_slots:
        Optional IDs of slots known to have poor visibility.

    Returns:
        Dictionary of contextual results by slot ID.
    """
    occluded_slots = set(occluded_slots or [])

    slot_shapes = {
        slot_id: Polygon(points)
        for slot_id, points in slots.items()
    }

    vehicle_shapes = [
        Polygon(points)
        for points in vehicles
    ]

    for shape in list(slot_shapes.values()) + vehicle_shapes:
        if not shape.is_valid or shape.area <= 0:
            raise ValueError("Invalid polygon geometry")

    # Identify the primary parking slot of each vehicle.
    primary_slots = []

    for vehicle in vehicle_shapes:
        overlaps = {
            slot_id: slot.intersection(vehicle).area
            for slot_id, slot in slot_shapes.items()
        }

        if not overlaps or max(overlaps.values()) <= 0:
            primary_slots.append(None)
        else:
            primary_slots.append(
                max(overlaps, key=overlaps.get)
            )

    combined = (
        unary_union(vehicle_shapes)
        if vehicle_shapes else None
    )

    results = {}

    for slot_id, slot in slot_shapes.items():
        consumed = (
            slot.intersection(combined).area
            if combined is not None else 0.0
        )

        remaining = (
            slot.difference(combined)
            if combined is not None else slot
        )

        occupied_fraction = consumed / slot.area

        usable_space = _has_usable_space(
            slot,
            remaining
        )

        direct_vehicles = [
            index
            for index, vehicle in enumerate(vehicle_shapes)
            if primary_slots[index] == slot_id
            and slot.intersection(vehicle).area > 0
        ]

        spillover_vehicles = [
            index
            for index, vehicle in enumerate(vehicle_shapes)
            if primary_slots[index] not in (None, slot_id)
            and slot.intersection(vehicle).area > 0
        ]

        # Direct occupancy takes priority.
        if direct_vehicles and occupied_fraction >= 0.35:
            label = "occupied_direct"
            status = "occupied"

        elif spillover_vehicles and not usable_space:
            label = "spillover_blocked"
            status = "occupied"

        elif direct_vehicles or spillover_vehicles:
            label = "partial_overlap_unknown"
            status = "unknown"

        elif slot_id in occluded_slots:
            label = "occluded_unknown"
            status = "unknown"

        else:
            label = "clear"
            status = "available"

        results[slot_id] = {
            "slot_id": slot_id,
            "slot_area": round(slot.area, 2),
            "consumed_area": round(consumed, 2),
            "remaining_area": round(remaining.area, 2),
            "occupied_fraction": round(
                occupied_fraction, 4
            ),
            "remaining_usable_flag": usable_space,
            "contextual_label": label,
            "initial_recognition": status,
            "direct_vehicles": direct_vehicles,
            "spillover_vehicles": spillover_vehicles
        }

    return results
