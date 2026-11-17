# End-to-End Development Roadmap & Dummy Data Plan

> **Module Root:** `ai_services/parking_occupancy/`  
> **Target Demo:** December 1–7, 2026 (subject to SENTRY integration & security gates)  
> **Current Test Baseline:** `40 passed` in `tests/unit/` and `tests/contract/` (using synthetic fixtures & `SyntheticDetector`)

---

## 1. Current Baseline vs. Next Development Stages

### Already Built & Verified in `src/parking_occupancy/`
- `capture.py` & `__main__.py`: Still/folder/video/stream ingestion, `--backend-mode mock|real`, `--synthetic-detector` fallback.
- `geometry.py` & `scripts/mark_polygons.py`: `PolygonMap`, `cv2.warpPerspective`, `cv2.perspectiveTransform` centroid projection, Shapely validation, exclusive `car_only | motorcycle_only` enforcement, resolution-mismatch → `UNKNOWN`.
- `inference.py`: CPU `YOLODetector` adapter + `SyntheticDetector` test double.
- `smoothing.py`: Asymmetric hysteresis (`n_to_occupied=3`, `n_to_available=5`), unknown interrupt guard.
- `freshness.py`: 30s fake-clock staleness, `boot_id` UUID4, per-space sequence monotonicity, strict fresh-frame recovery.
- `events.py` & `backend_client.py`: `CANDIDATE-2026-10-30.2` payloads (`correlation_id: null`, nullable `unknown` confidence), mock sink, bounded retry queue.
- `scripts/evaluate.py` & `tests/fixtures/`: Confusion matrix / false-available evaluator + synthetic blank/partial/full frames.

### Target Modules to Build Next (From This `docs/` Spec)
1. **Dual-Angle Geometry Enhancements (`geometry.py` + dual dummy configs):**
   - Ground-contact footprint polygon projection for `cctv_oblique` vs. full-box projection for `optimized_angle`.
   - Inverse homography ($H^{-1}$) helper for sightline shadow mapping.
   - Tracked dummy camera/polygon configs for both angle types.
2. **Adaptive Slot Mapping Module (`adaptive_mapping.py` *(new)*):**
   - Anchor-guided micro-drift compensation ($\Delta H$) with strict moved-camera $\tau_{\text{drift}}$ fault cutoff to `UNKNOWN`.
   - Row-corridor auto-partitioning & paint-line snapping for unlined or faint concrete zones (`strict_stalls` vs. `adaptive_curb`).
   - Effective usable polygon adjustment ($P_{\text{eff}}$) and per-slot depth distortion factor ($J_i$).
3. **Low-Trust Supporting Detector Calibration (`inference.py`):**
   - Configurable lower confidence proposal gate (`0.25`) + uncertainty margin metadata per `camera_angle_type`.
4. **Contextual Awareness Module (`contextual.py` *(new)*):**
   - Vehicle area consumed ($A_{\text{consumed}}$), remaining free area ($A_{\text{rem}}$), contiguous usable-area validation, spillover blocking, and foreground-to-rear sightline occlusion ($V_{\text{vis}}$).
5. **Cognitive Fusion AI Model (`fusion_model.py` *(new)*):**
   - Connects `slot_number`, `slot_area`, `initial_recognition`, `confidence_interval`, **`one_minute_mean_confidence`** (60s sliding window), and **`final_call`**.
6. **Expanded Unit, Integration & Evaluation Suites:**
   - Dummy 60-second time-series scenarios testing CCTV vs. optimized angle, micro-drift vs. camera-bump, unlined row partitioning, spillover, occlusion, low-confidence YOLO, 1-minute mean stability, and stale transitions.

---

## 2. Ordered Build Roadmap

| Step | Module / Files | Concrete Tasks | Acceptance Check (Dummy Data) |
|---|---|---|---|
| **Step 1** | `configs/cameras/`, `configs/polygons/`, `tests/fixtures/` | Create two dummy calibration pairs: `dummy_cctv_oblique` (skewed perspective) & `dummy_optimized_angle` (top-down); include `slot_number` alongside `space_id`. | `PolygonMap.from_json` loads both; `warp_to_bev` aligns skewed dummy bays into equal-area BEV rectangles. |
| **Step 2** | `geometry.py` | Add `project_detection_footprint()` (bottom quad for CCTV vs. full quad for optimized) + `slot_area` & `adjacent_slots` metadata + $H^{-1}$ projection. | Unit tests prove tall-vehicle centroid offset on oblique CCTV no longer lands in rear slot. |
| **Step 2B** | `adaptive_mapping.py` *(new)* | Implement anchor micro-drift $\Delta H$, unlined row corridor subdivision (`strict_stalls` & `adaptive_curb`), effective polygons $P_{\text{eff}}$, and depth stretch $J_i$. | Unit tests verify $\pm 5\text{ px}$ drift recovery, $> \tau_{\text{drift}}$ forced `UNKNOWN`, and unlined 4-bay corridor partitioning. |
| **Step 3** | `inference.py` | Expose low-confidence proposal floor and uncertainty metadata; ensure `person` class is excluded and missing weights fail cleanly. | `SyntheticDetector` feeds weak scores (`0.25–0.40`) without dropping candidates. |
| **Step 4** | `contextual.py` *(new)* | Implement `ContextualAssessor`: overlap area $\rho_i$, remaining area $A_{\text{rem}, i}$, usable-fit check, spillover detection, and sightline occlusion shadow ($V_{\text{vis}}$). | Unit tests pass for spillover blocking, rear-slot occlusion (`UNKNOWN`), and second-car-in-remaining-area scenarios. |
| **Step 5** | `fusion_model.py` *(new)* | Implement `CognitiveFusionModel`: fuse weighted signals (`w_area`, `w_ctx`, lower `w_det`), compute `confidence_interval` using $J_i$, maintain 60s sliding window `one_minute_mean_confidence`, emit `final_call`. | 60-second fake-clock test passes: passing car ignored, 2s YOLO drop smoothed, occlusion/stale forces `UNKNOWN`. |
| **Step 6** | `__main__.py`, `smoothing.py`, `freshness.py`, `events.py` | Wire `AdaptiveSlotMapper` → `ContextualAssessor` → `CognitiveFusionModel` → `StateSmoother` → `FreshnessTracker` → `EventEmitter` while keeping wire `occupancy_observation` strictly contract-compliant. | `python -m parking_occupancy --config ... --backend-mode mock` logs full cognitive fields locally and emits valid `CANDIDATE-2026-10-30.2` events. |
| **Step 7** | `scripts/evaluate.py`, `tests/integration/` | Extend `evaluate.py` to output confusion matrix, false-available rate, `unknown` rate, transition latency, and 1-min mean convergence across both dummy angle datasets. | Reproducible report generated locally on synthetic scenes; zero real campus data committed. |

---

## 3. Dummy Data Design Specification

Because live camera streams and backend endpoints are **BLOCKED** pending authentication/privacy/contract approvals, every module is built and verified against **explicitly dummy synthetic datasets**:

| Dummy Dataset | Purpose | Contents |
|---|---|---|
| **Dummy Set A: `cctv_oblique_sim`** | Tests Module 1 & 2 under heavy perspective | Synthetic oblique images generated by applying $H^{-1}$ to a clean 6-bay floorplan + simulated skewed vehicle boxes, foreground vans casting shadows, and straddling vehicles. |
| **Dummy Set B: `optimized_angle_sim`** | Tests Module 1 & 2 under clean angle | Near-overhead synthetic 6-bay grid (`4` `car_only`, `2` `motorcycle_only`) with clean bounding boxes and exact ground-truth area ratios. |
| **Dummy Set C: `60s_timeseries_sim`** | Tests Module 3 (1-min mean & CI) | Fake-clock JSON/scripted sequences over $t = 0\text{s}..180\text{s}$ simulating: arrival, 2s detector dropout, passing traffic, neighbor spillover, tree/van occlusion, camera disconnect (`>30s` stale), and reconnect (`new boot_id`). |

---

## 4. Alignment With SENTRY Canonical Phases & Contracts

- **SENTRY Phases 1–10:** This roadmap directly enriches SENTRY Phases 2–9 inside `ai_services/parking_occupancy/` without modifying `backend/`, `database/`, shared schemas, or other teams' directories.
- **No Unapproved Wire Changes:** New internal fields (`slot_number`, `slot_area`, `initial_recognition`, `confidence_interval`, `one_minute_mean_confidence`, `final_call`) live in the cognitive fusion layer and diagnostic logs/reports; the outbound HTTP payload in `events.py` preserves exact `CANDIDATE-2026-10-30.2` fields until a formal API-change review approves extensions.
- **Safety Gates Respected:** `BACKEND_MODE=real` stays disabled/blocked until approved contracts, device scopes, TLS, and SEC-01/SEC-02 clearance are confirmed.
