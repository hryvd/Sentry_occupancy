"""Generate per-slot synthetic BEV geometry training rows from SENTRY PolygonMap.

Usage (from repository root):
 python scripts/generate_contextual_geometry_data.py --geometry configs/polygons/YOUR_FILE.json

These are synthetic rule-labeled examples; not independent ground truth.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from shapely.affinity import rotate, translate
from shapely.geometry import box
from shapely.ops import unary_union

from parking_occupancy.geometry import PolygonMap

FIELDNAMES = [
    'session_id', 'frame_index', 'space_id', 'lot_code', 'compatibility',
    'scenario', 'slot_area', 'consumed_area', 'remaining_area',
    'occupied_fraction', 'visible_area_ratio', 'neighbor_overlap_fraction',
    'contextual_label', 'initial_recognition',
]


def make_vehicle(slot, scenario: str, rng: random.Random):
    """Create artificial BEV footprint around a registered slot's bounds."""
    minx, miny, maxx, maxy = slot.bounds
    w, h = maxx - minx, maxy - miny
    cx, cy = slot.centroid.x, slot.centroid.y
    if scenario == 'clear' or scenario == 'occluded_unknown':
        return []
    if scenario == 'occupied_direct':
        vehicle = box(cx - .39*w, cy - .40*h, cx + .39*w, cy + .40*h)
    elif scenario == 'partial_overlap':
        vehicle = box(cx + .30*w, cy - .35*h, cx + .95*w, cy + .35*h)
    elif scenario == 'spillover':
        vehicle = box(cx + .10*w, cy - .47*h, cx + 1.15*w, cy + .47*h)
    elif scenario == 'two_vehicles':
        a = box(cx - .43*w, cy - .40*h, cx - .05*w, cy + .40*h)
        b = box(cx + .05*w, cy - .40*h, cx + .43*w, cy + .40*h)
        return [a, b]
    else:
        raise ValueError(scenario)
    # Small controlled jitter to diversify training examples, without changing the slot geometry
    return [rotate(translate(vehicle, xoff=rng.uniform(-.025,.025)*w,
                              yoff=rng.uniform(-.025,.025)*h),
                   rng.uniform(-3,3), origin=(cx,cy))]


def assess(slot, vehicles, visibility: float, scenario: str):
    union = unary_union(vehicles) if vehicles else None
    consumed = slot.intersection(union).area if union is not None else 0.0
    area = slot.area
    rho = max(0.0, min(1.0, consumed / area))
    remaining = max(0.0, area - consumed)
    if rho >= .35:
        label, status = 'occupied_direct', 'occupied'
    elif visibility < .45:
        label, status = 'occluded_unknown', 'unknown'
    elif rho < .10 and visibility >= .70:
        label, status = 'clear', 'available'
    else:
        # Geometry alone cannot infer a blocked adjacent slot or whether a car can fit.
        label, status = 'ambiguous_overlap', 'unknown'
    return consumed, remaining, rho, label, status


def generate(geometry_path: Path, output_dir: Path, sessions: int, seed: int):
    poly_map = PolygonMap.from_json(geometry_path)
    if not poly_map.spaces:
        raise ValueError('Geometry JSON has no registered spaces')
    rng = random.Random(seed)
    scenarios = ('clear', 'occupied_direct', 'partial_overlap', 'spillover',
                 'two_vehicles', 'occluded_unknown')
    rows = []
    for session in range(sessions):
        for space in poly_map.spaces:
            slot = space._shapely  # Use validated polygon from SENTRY's actual Geometry model
            for step, scenario in enumerate(scenarios):
                vehicles = make_vehicle(slot, scenario, rng)
                visibility = rng.uniform(.1,.35) if scenario == 'occluded_unknown' else rng.uniform(.75,1.)
                consumed, remaining, rho, label, status = assess(slot, vehicles, visibility, scenario)
                rows.append({
                    'session_id': f'session_{session:04d}', 'frame_index': step,
                    'space_id': space.space_id, 'lot_code': space.lot_code,
                    'compatibility': space.compatibility.value,
                    'scenario': scenario, 'slot_area': round(slot.area,3),
                    'consumed_area': round(consumed,3),
                    'remaining_area': round(remaining,3),
                    'occupied_fraction': round(rho,6),
                    'visible_area_ratio': round(visibility,6),
                    # 0 here means neighbor spillover has not yet been validated
                    'neighbor_overlap_fraction': 0.0,
                    'contextual_label': label, 'initial_recognition': status,
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    # session-based split (prevents near-duplicate rows from same synthetic session leaking across splits)
    cut_train = int(sessions*.7)
    cut_val = int(sessions*.85)
    groups = {'train': [], 'val': [], 'test': []}
    for row in rows:
        session = int(row['session_id'].split('_')[1])
        split = 'train' if session < cut_train else 'val' if session < cut_val else 'test'
        groups[split].append(row)
    for name, records in [('all_records',rows), *groups.items()]:
        with (output_dir/f'{name}.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(records)
    print(f'Loaded {len(poly_map.spaces)} registered spaces from {geometry_path}')
    print(f'Generated {len(rows)} synthetic records: ' + ', '.join(f'{k}={len(v)}' for k,v in groups.items()))
    print(f'Saved CSV files to {output_dir}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('data/contextual/geometry_dataset'))
    parser.add_argument('--sessions', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.sessions < 7:
        parser.error('--sessions must be at least 7 for meaningful train/val/test splits')
    generate(args.geometry, args.out, args.sessions, args.seed)


if __name__ == '__main__':
    main()
