
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
