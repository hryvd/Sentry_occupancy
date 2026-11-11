# SENTRY — Parking Occupancy AI: 3-View Warp & Occupancy Detection Plan

> **Document version:** 1.0 | **Updated:** 2026-11-11 | **Team:** Occupancy AI
>
> This plan covers the occupancy team's scope only: `ai_services/parking_occupancy/`.
> It does not authorize backend changes, assignment decisions, or real camera credentials.
> All features use synthetic inputs and mock ports until approved integration gates are met.

---

## 1. The 3 Camera Views — BSU Campus Parking

Three actual parking area cameras capture angled overhead (perspective) images of different lots.
Each must be **independently warped** to a bird's-eye / top-down reference plane before
polygon-based occupancy can be reliably assessed.

### View A — Dense Lot (Lots A/B)
- **Camera position:** Upper floor, moderate angle (~40–55°)
- **Visible spaces:** ~20–25 car spaces, mixed orientation (perpendicular + angled rows)
- **Warp target size:** 600×500 px reference canvas
- **Challenges:**
  - Vehicles at the near edge are very large (foreshortening inversion)
  - Rear-row vehicles partially occluded by front-row rooftops
  - Pedestrian movement in frame (must not be classified as a vehicle)
- **Polygon strategy:** One polygon per bay; tight-fit quadrilateral ROIs (not rectangles) to handle angled stalls

### View B — Wide Lot (Lots B/C)
- **Camera position:** Upper floor, wider field, shallower angle (~30–45°)
- **Visible spaces:** ~25–30 car spaces with lane markings visible
- **Warp target size:** 800×600 px reference canvas
- **Challenges:**
  - Strong scale variation — far vehicles appear very small
  - Right edge clips into greenery/building
  - White van occludes behind row
- **Polygon strategy:** Use lane marking corners as calibration anchors; group into 3–4 rows; mark `unknown` for occluded far-right corner

### View C — Side Angle / Near Building (Lots C/D)
- **Camera position:** Upper floor, significant lateral angle, broad view
- **Visible spaces:** ~10–15 car spaces near building + driveway/access lane
- **Warp target size:** 700×500 px reference canvas
- **Challenges:**
  - Strong lateral perspective (most difficult warp)
  - Trees/planters occlude middle zones → mark as `unknown`
  - Driveway lane must be **excluded** (not a parking space)
  - Circular fountain/garden area in foreground is not a parking area
- **Polygon strategy:** Only annotate confirmed bay areas; leave occluded tree zones unpolygoned (auto-`unknown`)

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│          Per-Camera Pipeline (runs for each of 3 views)    │
│                                                            │
│  Camera Frame ──► YOLOv8 Detector ──► Project to BEV      │
│  (original)        (original coords)   via H matrix        │
│                                            │               │
│                                     Polygon Containment    │
│                                     (geometry.py)          │
│                                            │               │
│                               available | occupied | unknown│
│                                            │               │
│                                  Temporal Smoother         │
│                                  (smoothing.py)            │
│                                            │               │
│                                  Freshness Tracker         │
│                                  (freshness.py)            │
└────────────────────────────────────────────────────────────┘
                              │
                     Multi-View Aggregator
                     (fuse overlapping zones)
                              │
                    Event Serializer
                    OCCUPANCY / HEARTBEAT
                              │
                    Backend Client
                    POST /api/v1/observations/occupancy
                    (mock in dev; real is BLOCKED)
```

---

## 3. The Warping (Homography) Approach

### What is a Homography Warp?

A **homography** is a 3×3 matrix H that maps any point (x, y) in the original
perspective image to a corresponding point (x', y') in the flat top-down reference plane:

```
[x']       [h00 h01 h02]   [x]
[y']  ~    [h10 h11 h12] × [y]
[1 ]       [h20 h21 h22]   [1]
```

This corrects the perspective distortion introduced by the camera's elevated,
angled mounting position, so parking spaces appear as regular rectangles (like a floor plan).

### Calibration Workflow (via `mark_polygons.py`)

```
1. Load reference camera frame (still image taken at calibration time)
        │
2. Click 4+ ground-plane points on the ORIGINAL image
        │
3. Enter matching coordinates in the BEV canvas
        │
4. Compute H = cv2.findHomography(src_pts, dst_pts, RANSAC)
        │
5. Apply cv2.warpPerspective() → preview BEV
   Validate: lane markings parallel? correct scale?
        │
6. Click each parking space boundary in the BEV image
        │
7. Export JSON: H matrix + src/dst points + space polygons + calibration_version
```

### Source Point Selection Strategy

| View | Recommended anchor features |
|------|------------------------------|
| **View A** | 4 corners of the clearly-visible concrete slab boundary + painted lot lines |
| **View B** | 4 corners of painted parking bay lines (white lane markings) |
| **View C** | Corner of driveway lane markings + building wall base corners |

**Key rule:** If the camera is physically moved or its angle changes, H becomes invalid.
All spaces for that camera become `unknown` until recalibrated (`freshness.py` enforces this
via `calibration_version` mismatch detection).

---

## 4. Detection Strategy

**Detect in the ORIGINAL image, project to BEV — not the other way around.**

```
Original Frame
     │
     ├──► YOLOv8 detects vehicles (car / motorcycle) in original pixel coords
     │         COCO class 2 = car, class 3 = motorcycle
     │
     ▼
Bounding-box centroids (cx, cy) in original coords
     │
     ├──► cv2.perspectiveTransform(centroid, H) → (bev_x, bev_y)
     │
     ▼
Test: does (bev_x, bev_y) fall inside space polygon?
     │
     ├── YES + compatible type → OCCUPIED (high confidence)
     ├── YES + wrong type     → OCCUPIED (flagged, lower confidence)
     └── NO                  → AVAILABLE (tentative; needs smoothing)
```

Detecting in the original image (not the warped one) avoids YOLO accuracy degradation
caused by perspective warp blur and distortion artifacts.

---

## 5. Occupancy States

| State | Meaning | Confidence |
|-------|---------|------------|
| `occupied` | Compatible vehicle detection confirmed in space polygon | 0.0–1.0 |
| `available` | No detection in polygon for N consecutive frames | 0.0–1.0 |
| `unknown` | Insufficient evidence, occlusion, stale/disconnected camera, calibration mismatch | `null` |

**Critical rules:**
- Absence of detections alone NEVER means available — smoothing must confirm
- Stale camera (no fresh frame > 30s) → ALL spaces on that camera → `unknown`
- `unknown` NEVER triggers backend to release active assignment/visit
- Driveway lanes and occluded areas have NO polygon → no classification

---

## 6. Temporal Smoothing (Hysteresis)

```
                N_OCC frames                    N_AVAIL frames
unknown ─────────────────────► occupied ◄────────────────────── available
   ▲                              │                                  ▲
   │◄──── any UNKNOWN evidence ───┤                                  │
   │          (resets counter)    │                                  │
   └─────────────────────────────►└────────────────────────────────►─┘
        N_AVAIL consecutive available                 N_OCC consecutive occupied
```

Default: `N_OCC = 3`, `N_AVAIL = 5` (configurable per camera YAML).
More evidence is required to declare a space vacant than occupied — prevents false vacancies from brief non-detections.

---

## 7. Freshness & Stale Detection

```
Every frame:
  last_frame_time = now()

If (now() - last_frame_time) > FRESHNESS_THRESHOLD_S:
  camera is STALE → all spaces → unknown

On camera reconnect:
  - New boot_id (UUID4 generated)
  - All per-space sequence numbers reset to 0
  - Stale flag cleared
  - Fresh frames (not heartbeat) restore coverage
```

**Heartbeat does NOT refresh stale space observations.**
Only genuine fresh frames with new pixel data restore coverage.

---

## 8. Event Schema (Provisional CANDIDATE-2026-10-30.2)

```json
{
  "schema_version": "1.0.0",
  "event_id": "<uuid4>",
  "event_type": "occupancy_observation",
  "occurred_at": "2026-11-11T15:00:00Z",
  "producer_id": "occupancy-lot-a-01",
  "correlation_id": null,
  "data": {
    "lot_code": "A",
    "space_id": "30000000-0000-4000-8000-000000000001",
    "physical_status": "occupied",
    "confidence": 0.94,
    "boot_id": "<uuid4>",
    "sequence_number": 17,
    "calibration_version": "lot-a-view1-polygons-1",
    "model_version": "occupancy-demo-1"
  }
}
```

**`correlation_id` is ALWAYS `null`** — occupancy observations are never tied to a specific access attempt.
**`confidence` is `null`** when `physical_status` is `"unknown"`.

---

## 9. Per-View Calibration Config Structure

```json
{
  "calibration_version": "lot-a-cam01-v1",
  "camera_id": "cam-lot-a-01",
  "lot_code": "A",
  "reference_resolution": [1280, 960],
  "bev_canvas_size": [600, 500],
  "homography_matrix": [[...3x3 floats...]],
  "src_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "dst_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "spaces": [
    {
      "space_id": "30000000-0000-4000-8000-000000000001",
      "lot_code": "A",
      "compatibility": "car_only",
      "polygon_bev": [[x,y], [x,y], [x,y], [x,y]]
    }
  ]
}
```

Three configs needed (one per view):
- `configs/polygons/lot_a_view1.json` — View A (Lots A/B)
- `configs/polygons/lot_b_view2.json` — View B (Lots B/C)
- `configs/polygons/lot_c_view3.json` — View C (Lots C/D) — driveway excluded

---

## 10. Key Technical Constraints

| Constraint | Decision |
|------------|----------|
| Camera calibration change | `freshness.py` detects `calibration_version` mismatch → all spaces `unknown` |
| Pedestrians in parking area | YOLO detects `person` (class 0) but NOT associated with any space polygon |
| Occluded space (tree/van) | Polygon still exists; detection absence alone ≠ available; smoother or → `unknown` |
| Wrong vehicle type in space | Report `occupied`, flag type mismatch; never reclassify space compatibility |
| Multi-view overlap | Aggregator picks fresher / higher-confidence observation per space |
| No identity inference | Detected box NEVER yields a user UUID — strictly backend business |
| Driveway lanes (View C) | No polygon drawn; driveways excluded from space config |
| Mixed spaces | NEVER created — `car_only` and `motorcycle_only` only |

---

## 11. Milestone Targets

| Date | Deliverable |
|------|-------------|
| **Nov 12** | Phase 1–3: environment, space config, `mark_polygons.py` working with synthetic BEV output for all 3 views |
| **Nov 19** | Phase 4–6: YOLO detection, occupancy classification, temporal smoothing; mock backend sink running |
| **Nov 24** | Phase 7–8: stale/unknown/reconnect; full event serialization; contract tests passing |
| **Nov 30** | Phase 9–10: evaluation report, mock conflict demo, handoff-ready |
| **Dec 1–7** | Demonstration scope declared; real-camera connection blocked until approved auth/TLS/device scope |

---

## 12. Synthetic Fixture Plan

| Fixture | Description | Used in |
|---------|-------------|---------|
| `example_empty.jpg` | Blank canvas (no vehicles) | Phase 1, 5, 6 |
| `example_full.jpg` | All bays occupied (colored rectangles) | Phase 5, 9 |
| `example_partial.jpg` | ~50% occupancy | Phase 5, 6, 9 |
| OpenCV-drawn synthetic scenes | Blank canvas + colored rectangles | Phase 3, 4 |

> **Warning:** Do NOT commit real campus camera frames with identifiable persons into Git.
> Use only clearly synthetic reference calibration images or heavily cropped/anonymized stills.

---

## 13. Integration Boundary Summary

```
Occupancy AI                        FastAPI Backend
─────────────────                   ───────────────────────────
Observes physical state             Owns assignment + allocation
Reports available/occupied/unknown  Decides conflicts
Never identifies occupant           Owns official visit/history
Never reserves/releases spaces      Owns parking assignment records
Never accesses PostgreSQL           Sole database writer
```
