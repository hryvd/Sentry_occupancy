# SENTRY — Parking Occupancy Detection Service

> **Team:** Occupancy AI  
> **Module root:** `ai_services/parking_occupancy/`  
> **Target demo:** December 1–7, 2026  
> **Status:** `READY_FOR_MOCK_DEVELOPMENT` — real camera integration BLOCKED (see below)

Estimates per-space physical occupancy for campus parking **Lots A, B, C, and D** using three camera views, perspective warping (homography), and YOLOv8 vehicle detection. Reports observations to the FastAPI backend via `POST /api/v1/observations/occupancy` (provisional candidate endpoint).

**This module is an observation reporter only.** It never reserves/releases spaces, identifies drivers, writes to PostgreSQL, or makes authorization decisions. Those are exclusively FastAPI backend responsibilities.

---

## Camera Views

| Camera | Lot coverage | Warp challenge |
|--------|-------------|----------------|
| `cam-lot-a-01` | Lot A/B — dense multi-row | Foreshortening on far rows, mixed orientations |
| `cam-lot-b-02` | Lot B/C — wide field | Strong scale variation, edge clipping |
| `cam-lot-c-03` | Lot C/D — side angle | Lateral perspective, tree/planter occlusion |

Each camera has an independent calibrated homography matrix `H` stored in `configs/polygons/`. Run `scripts/mark_polygons.py` to generate or update polygon configs.

---

## Quick Start (Mock Mode)

```powershell
# 1. Create virtual environment (Python 3.10+)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Generate synthetic test frames
python tests/fixtures/generate_synthetic_frames.py

# 4. Run all tests (no GPU or camera required)
python -m pytest tests/ -v

# 5. Run against a single still image in mock mode
python -m parking_occupancy --config configs/cameras/example.yaml `
    --still tests/fixtures/example_partial.jpg `
    --backend-mode mock

# 6. Annotate polygons for a new camera view
python scripts/mark_polygons.py `
    --image <calibration_still.jpg> `
    --camera-id cam-lot-a-01 `
    --lot A `
    --output configs/polygons/lot_a_view1.json
```

---

## Architecture — 3-View Warp Pipeline

```
Camera Frame (Original Image)
        │
        ▼
  YOLOv8 Detector          ← Detects cars + motorcycles in ORIGINAL coords
  (inference.py)
        │
        ▼
  Project centroids          ← cv2.perspectiveTransform() through H
  to BEV via H
  (geometry.py)
        │
        ▼
  Polygon containment        ← Which space polygon contains each detection?
  classification
  (geometry.py)
        │  available | occupied | unknown (per space, per frame)
        ▼
  Temporal smoother          ← N consecutive frames to confirm transition
  (smoothing.py)             ← More evidence needed to go available than occupied
        │
        ▼
  Freshness tracker          ← Stale detection, boot_id, sequence numbers
  (freshness.py)             ← No frame for >30s → all spaces unknown
        │
        ▼
  Event emitter              ← Emit on state change OR periodic refresh
  (events.py)
        │  OCCUPANCY / HEARTBEAT (correlation_id always null)
        ▼
  Backend client             ← Mock sink (dev) or real POST (gated)
  (backend_client.py)
```

**Why detect in original image, not warped?**  
Warping blurs/distorts vehicle shapes and degrades YOLO accuracy. We detect in the original perspective image, then project detection centroids through the homography `H` into BEV space for polygon testing.

---

## Homography / Perspective Warp

Each camera view is calibrated once using `mark_polygons.py`:
1. Select 4+ ground-plane correspondence points on the original camera image
2. Enter matching coordinates in the bird's-eye (top-down) reference canvas
3. Tool computes `H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)`
4. Preview the warped BEV — confirm lane markings are parallel
5. Annotate each parking space as a polygon in BEV space
6. Config JSON saved with embedded `H` matrix + space polygons

If the camera is physically moved, `calibration_version` mismatch is detected and all spaces on that camera become `unknown` until recalibrated.

---

## File Structure

```
ai_services/parking_occupancy/
├── src/parking_occupancy/
│   ├── __init__.py          Version declaration
│   ├── __main__.py          CLI entry point (--config --still --video --folder)
│   ├── capture.py           Frame acquisition (still / folder / video / stream)
│   ├── inference.py         YOLOv8 detector adapter + SyntheticDetector (tests)
│   ├── geometry.py          Homography warp, polygon map, occupancy classifier
│   ├── smoothing.py         Hysteresis state machine (per-space temporal smoothing)
│   ├── freshness.py         Stale detection, boot_id, sequence tracking
│   ├── events.py            OCCUPANCY + HEARTBEAT payload builders
│   └── backend_client.py    Mock/real POST transport + bounded retry queue
├── configs/
│   ├── cameras/example.yaml      Camera config template
│   └── polygons/example.json     Synthetic polygon map (identity H, 3 spaces)
├── scripts/
│   ├── mark_polygons.py     Interactive warp calibration + polygon annotation tool
│   └── evaluate.py          Per-state confusion matrix evaluation
├── models/
│   ├── README.md            Weight download instructions
│   └── manifest.example.json
├── tests/
│   ├── unit/test_geometry.py    Homography, polygon validation, classify()
│   ├── unit/test_smoothing.py   Hysteresis, flicker, unknown evidence
│   ├── unit/test_freshness.py   Stale, reconnect, boot_id, sequences
│   ├── contract/test_events.py  Schema, null correlation_id, retry preservation
│   └── fixtures/
│       ├── annotations.json     Ground-truth for synthetic frames
│       └── generate_synthetic_frames.py
├── pyproject.toml
├── .env.example
└── .gitignore
```

---

## Configuration

Copy `.env.example` → `.env` and fill in:

| Variable | Description |
|----------|-------------|
| `BACKEND_MODE` | `mock` (development) or `real` (requires auth/TLS approval) |
| `CAMERA_SOURCE` | RTSP URL, webcam index, or leave blank for `--still`/`--video` |
| `POLYGON_CONFIG` | Path to polygon JSON for this camera |
| `MODEL_PATH` | Path to `yolov8n.pt` weights |
| `DEVICE_ID` | Camera device identifier (e.g., `occupancy-lot-a-01`) |
| `LOT_CODE` | Parking lot (A/B/C/D) |

Camera configs live in `configs/cameras/`. Polygon configs in `configs/polygons/`. Private overrides with real RTSP URLs or credentials are **gitignored**.

---

## Implementation Phases

| Phase | Focus | Status |
|-------|-------|--------|
| 1. Environment & capture | Python env, capture modes, synthetic frames | ✅ Scaffolded |
| 2. Space config | Synthetic space UUIDs, car/moto compatibility | ✅ Scaffolded |
| 3. Polygon annotation | `mark_polygons.py`, homography, BEV preview | ✅ Scaffolded |
| 4. Vehicle detection | YOLOv8 adapter, SyntheticDetector, model manifest | ✅ Scaffolded |
| 5. Occupancy classification | Polygon containment, type compat, BEV projection | ✅ Scaffolded |
| 6. Temporal smoothing | Hysteresis, consecutive frames, flicker | ✅ Scaffolded |
| 7. Unknown/stale/reconnect | Freshness, boot_id, sequence tracking | ✅ Scaffolded |
| 8. Backend events & mock | OCCUPANCY/HEARTBEAT schema, retry queue | ✅ Scaffolded |
| 9. Evaluation | Confusion matrix, evaluate.py, synthetic fixtures | ✅ Scaffolded |
| 10. Integration & handoff | Mock conflict demo, real adapter gated | 🔒 BLOCKED |

---

## Integration Status

| Interface | Status |
|-----------|--------|
| `GET /health/live`, `GET /health/ready` | `IMPLEMENTED_AND_AVAILABLE` |
| `POST /api/v1/observations/occupancy` | `PROPOSED_PENDING_APPROVAL` — use mock |
| `POST /api/v1/observations/heartbeat` | `PROPOSED_PENDING_APPROVAL` — use mock |
| Availability / assignment / WS reads | `NOT_IMPLEMENTED` |

Real adapter requires: approved auth/TLS, device scopes, implemented backend handlers, and cleared SEC-01/SEC-02 gates.

---

## Testing

```powershell
# Full test suite (no camera, no GPU needed)
python -m pytest tests/ -v --tb=short

# Unit tests only
python -m pytest tests/unit/ -v

# Contract tests
python -m pytest tests/contract/ -v

# With coverage
python -m pytest tests/ --cov=parking_occupancy --cov-report=term-missing
```

**Key test scenarios covered:**
- Polygon validation (degenerate, duplicate IDs, out-of-bounds, mixed compat rejected)
- Classify: occupied, available, resolution mismatch → unknown, wrong-type vehicle → still occupied
- Smoothing: hysteresis boundaries, flicker suppression, unknown resets pending
- Freshness: stale transitions, reconnect, boot_id change, sequences reset
- Events: `correlation_id` always null, `confidence` null for unknown, retry preserves event_id

---

## Submission Checklist

- [ ] Clean install: `pip install -e ".[dev]"` succeeds with no errors
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] `--help` works: `python -m parking_occupancy --help`
- [ ] Mock run works: `python -m parking_occupancy --config configs/cameras/example.yaml --still tests/fixtures/example_partial.jpg --backend-mode mock`
- [ ] No real camera URLs, credentials, or private images committed
- [ ] Model weights absent (gitignored); manifest.example.json present
- [ ] Real adapter marked BLOCKED in module README

**Owner / Backup:** _[Assign names before submission]_  
**Accepted contract version:** CANDIDATE-2026-10-30.2 (provisional — not yet approved)
