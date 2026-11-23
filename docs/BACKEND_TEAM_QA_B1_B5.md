# Backend Integration Q&A — Campus Lot B (`B1–B5`, Focus Areas `B2, B3, B4`)

> **Scope Summary:** Campus parking sensors cover **Lot `B` sub-areas `B1–B5`** (`lot_code: "B"`). Active calibration and engineering coverage center on **`B2`, `B3`, and `B4`** (`B2` and `B4` are calibrated and running across **49 bays**; `B3` is in active rollout awaiting its fixed camera reference frame; `B1` and `B5` are planned). Lots `A`, `C`, and `D` are outside this service scope.
>
> **Architecture Alignment:** Backend HTTP `POST` occupancy messages are driven directly from our 4-model fusion CSV (`area,slot_id,initial_state,model_1,model_2,model_3,model_4,mean,final_state,final_id`). A lightweight continuous change-monitor suppresses occupancy `POST`s when there is no significant vehicle change (sending only `60s` heartbeats) and dispatches **at least 2 updated full final-state payloads per minute** (`<= 30s` cadence across active cameras) during high-change activity.

---

## 1. Slots per Lot (`B1–B5` vs. Active `B2, B3, B4`) & Slot-ID Mapping

### Campus Coverage Overview

All vision-monitored spaces in this rollout belong to campus **Lot `B` (`lot_code: "B"`)**, partitioned operationally into five zones **`B1–B5`**:

| Sub-Area | Campus Group | Status | Calibrated Slot Count | Producer / `device_id` | Polygon & Camera Configs |
|---|---|---|---|---|---|
| **`B2`** | `B1–B5` | **Calibrated & Active** | **22** (`L1–L11`, `R1–R11`) | `occupancy-area-b2-01` | [`area_b2.json`](../configs/polygons/area_b2.json) / [`area_b2.yaml`](../configs/cameras/area_b2.yaml) |
| **`B3`** | `B1–B5` | **Understood — Active Rollout** *(pending reference image)* | TBD on camera frame lock | `occupancy-area-b3-01` *(planned)* | Planned (`area_b3.json` / `area_b3.yaml`) |
| **`B4`** | `B1–B5` | **Calibrated & Active** | **27** (`L1–L8`, `R1–R10`, `MB1–MB9`) | `occupancy-area-b4-01` | [`area_b4.json`](../configs/polygons/area_b4.json) / [`area_b4.yaml`](../configs/cameras/area_b4.yaml) |
| **`B1`, `B5`** | `B1–B5` | **Planned Expansion** | TBD | `occupancy-area-b1-01`, `occupancy-area-b5-01` | Planned |
| `A`, `C`, `D` | Outside scope | Not covered by this service | — | — | — |

### Do You Need a Mapping for Backend `B-P7AB12CD34` Names?

**Yes — a mapping table is required.**

- Inside the AI service and CSV output (`lot_b2_b4_occupancy.csv`), slots are identified human-readably by **`(area, slot_id)`** — for example `("B2", "L1")` or `("B4", "MB1")`. Note that `"L1"` appears in both `B2` and `B4`, so `area` is always required as part of the natural key.
- Over HTTP (`CANDIDATE-2026-10-30.2`), `data.space_id` carries a UUID string read from each polygon config. Today our configs use deterministic provisional UUIDs (`30000000-0000-4000-8000-000000000101`..`0122` for `B2`, and `30000000-0000-4000-8000-000000000201`..`0227` for `B4`) alongside `lot_code: "B"`.
- **Zero-code swap:** Once the backend generates its canonical slot records (your `B-P...` display codes and database UUIDs), share the `(area, slot_id) -> (space_id UUID, backend_code)` seed mapping and we will update `space_id` directly in `configs/polygons/area_b*.json` without changing any pipeline code.

### Complete Active Slot Inventory — Lot `B2` (22 Bays) & Lot `B4` (27 Bays)

<details>
<summary>Expand full 49-slot mapping table (Area B2: 22 slots, Area B4: 27 slots)</summary>

#### Area B2 (`camera_id: cam-area-b2-01`, `producer_id: occupancy-area-b2-01`, `calibration_version: area-b2-4model-v1`)

| Area | Local `slot_id` | Zone | Provisional `space_id` (`CANDIDATE-2026-10-30.2`) | `compatibility` | Backend Display Code |
|---|---|---|---|---|---|
| `B2` | `L1` | `left` | `30000000-0000-4000-8000-000000000101` | `car_only` | TBD (`B-P...`) |
| `B2` | `L2` | `left` | `30000000-0000-4000-8000-000000000102` | `car_only` | TBD (`B-P...`) |
| `B2` | `L3` | `left` | `30000000-0000-4000-8000-000000000103` | `car_only` | TBD (`B-P...`) |
| `B2` | `L4` | `left` | `30000000-0000-4000-8000-000000000104` | `car_only` | TBD (`B-P...`) |
| `B2` | `L5` | `left` | `30000000-0000-4000-8000-000000000105` | `car_only` | TBD (`B-P...`) |
| `B2` | `L6` | `left` | `30000000-0000-4000-8000-000000000106` | `car_only` | TBD (`B-P...`) |
| `B2` | `L7` | `left` | `30000000-0000-4000-8000-000000000107` | `car_only` | TBD (`B-P...`) |
| `B2` | `L8` | `left` | `30000000-0000-4000-8000-000000000108` | `car_only` | TBD (`B-P...`) |
| `B2` | `L9` | `left` | `30000000-0000-4000-8000-000000000109` | `car_only` | TBD (`B-P...`) |
| `B2` | `L10` | `left` | `30000000-0000-4000-8000-000000000110` | `car_only` | TBD (`B-P...`) |
| `B2` | `L11` | `left` | `30000000-0000-4000-8000-000000000111` | `car_only` | TBD (`B-P...`) |
| `B2` | `R1` | `right` | `30000000-0000-4000-8000-000000000112` | `car_only` | TBD (`B-P...`) |
| `B2` | `R2` | `right` | `30000000-0000-4000-8000-000000000113` | `car_only` | TBD (`B-P...`) |
| `B2` | `R3` | `right` | `30000000-0000-4000-8000-000000000114` | `car_only` | TBD (`B-P...`) |
| `B2` | `R4` | `right` | `30000000-0000-4000-8000-000000000115` | `car_only` | TBD (`B-P...`) |
| `B2` | `R5` | `right` | `30000000-0000-4000-8000-000000000116` | `car_only` | TBD (`B-P...`) |
| `B2` | `R6` | `right` | `30000000-0000-4000-8000-000000000117` | `car_only` | TBD (`B-P...`) |
| `B2` | `R7` | `right` | `30000000-0000-4000-8000-000000000118` | `car_only` | TBD (`B-P...`) |
| `B2` | `R8` | `right` | `30000000-0000-4000-8000-000000000119` | `car_only` | TBD (`B-P...`) |
| `B2` | `R9` | `right` | `30000000-0000-4000-8000-000000000120` | `car_only` | TBD (`B-P...`) |
| `B2` | `R10` | `right` | `30000000-0000-4000-8000-000000000121` | `car_only` | TBD (`B-P...`) |
| `B2` | `R11` | `right` | `30000000-0000-4000-8000-000000000122` | `car_only` | TBD (`B-P...`) |

#### Area B4 (`camera_id: cam-area-b4-01`, `producer_id: occupancy-area-b4-01`, `calibration_version: area-b4-4model-v1`)

| Area | Local `slot_id` | Zone | Provisional `space_id` (`CANDIDATE-2026-10-30.2`) | `compatibility` | Backend Display Code |
|---|---|---|---|---|---|
| `B4` | `L1` | `left` | `30000000-0000-4000-8000-000000000201` | `car_only` | TBD (`B-P...`) |
| `B4` | `L2` | `left` | `30000000-0000-4000-8000-000000000202` | `car_only` | TBD (`B-P...`) |
| `B4` | `L3` | `left` | `30000000-0000-4000-8000-000000000203` | `car_only` | TBD (`B-P...`) |
| `B4` | `L4` | `left` | `30000000-0000-4000-8000-000000000204` | `car_only` | TBD (`B-P...`) |
| `B4` | `L5` | `left` | `30000000-0000-4000-8000-000000000205` | `car_only` | TBD (`B-P...`) |
| `B4` | `L6` | `left` | `30000000-0000-4000-8000-000000000206` | `car_only` | TBD (`B-P...`) |
| `B4` | `L7` | `left` | `30000000-0000-4000-8000-000000000207` | `car_only` | TBD (`B-P...`) |
| `B4` | `L8` | `left` | `30000000-0000-4000-8000-000000000208` | `car_only` | TBD (`B-P...`) |
| `B4` | `R1` | `right` | `30000000-0000-4000-8000-000000000209` | `car_only` | TBD (`B-P...`) |
| `B4` | `R2` | `right` | `30000000-0000-4000-8000-000000000210` | `car_only` | TBD (`B-P...`) |
| `B4` | `R3` | `right` | `30000000-0000-4000-8000-000000000211` | `car_only` | TBD (`B-P...`) |
| `B4` | `R4` | `right` | `30000000-0000-4000-8000-000000000212` | `car_only` | TBD (`B-P...`) |
| `B4` | `R5` | `right` | `30000000-0000-4000-8000-000000000213` | `car_only` | TBD (`B-P...`) |
| `B4` | `R6` | `right` | `30000000-0000-4000-8000-000000000214` | `car_only` | TBD (`B-P...`) |
| `B4` | `R7` | `right` | `30000000-0000-4000-8000-000000000215` | `car_only` | TBD (`B-P...`) |
| `B4` | `R8` | `right` | `30000000-0000-4000-8000-000000000216` | `car_only` | TBD (`B-P...`) |
| `B4` | `R9` | `right` | `30000000-0000-4000-8000-000000000217` | `car_only` | TBD (`B-P...`) |
| `B4` | `R10` | `right` | `30000000-0000-4000-8000-000000000218` | `car_only` | TBD (`B-P...`) |
| `B4` | `MB1` | `moto` | `30000000-0000-4000-8000-000000000219` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB2` | `moto` | `30000000-0000-4000-8000-000000000220` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB3` | `moto` | `30000000-0000-4000-8000-000000000221` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB4` | `moto` | `30000000-0000-4000-8000-000000000222` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB5` | `moto` | `30000000-0000-4000-8000-000000000223` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB6` | `moto` | `30000000-0000-4000-8000-000000000224` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB7` | `moto` | `30000000-0000-4000-8000-000000000225` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB8` | `moto` | `30000000-0000-4000-8000-000000000226` | `motorcycle_only` | TBD (`B-P...`) |
| `B4` | `MB9` | `moto` | `30000000-0000-4000-8000-000000000227` | `motorcycle_only` | TBD (`B-P...`) |

</details>

---

## 2. Slot Types (`car_only` / `motorcycle_only`) & Admin/Faculty Reservations

- **Strict two-type enum (`car_only | motorcycle_only`):** Every parking space in `src/parking_occupancy/geometry.py` and `configs/polygons/*.json` is explicitly typed as either **`car_only`** or **`motorcycle_only`**. We **do not support `mixed` slots** so stall polygons and capacity rules remain unambiguous:
  - **Area `B2` (22 total):** `22 car_only`, `0 motorcycle_only`, `0 mixed`.
  - **Area `B4` (27 total):** `18 car_only` (`L1–L8`, `R1–R10`), `9 motorcycle_only` (`MB1–MB9`), `0 mixed`.
  - **`B3` / `B1` / `B5`:** Will follow the same strict `car_only | motorcycle_only` classification upon calibration.
- **Admin / Faculty Reservations (`Lot A` & `B1–B5` policy):**
  - **Lot `A` is outside our `B1–B5` scope.**
  - Within `B1–B5`, the vision service emits **physical occupancy only** (`occupied | available | unknown`) and **never handles driver identities, plates, permits, or role reservations** (`SEC-02`). Any admin/faculty or reserved-bay flags belong strictly in the backend space metadata table and are overlaid on top of physical occupancy in the backend/UI.

---

## 3. Cameras, Slot Coverage, `device_id` Mapping, & Multi-Camera Execution

### One Camera and One `device_id` per Area

Our topology assigns **1 fixed overhead/oblique camera per sub-area** and **1 unique `device_id` (`producer_id`) per camera**, matching the backend requirement of **one API key per device** (`SEC-01`):

| Camera ID | `device_id` (`producer_id`) | Areas & Slots Covered | Auth Key Scope | Status |
|---|---|---|---|---|
| `cam-area-b2-01` | `occupancy-area-b2-01` | All **22** bays in Area `B2` (`...0101`–`...0122`) | 1 API key (`occupancy-area-b2-01`) | Active |
| `cam-area-b4-01` | `occupancy-area-b4-01` | All **27** bays in Area `B4` (`...0201`–`...0227`) | 1 API key (`occupancy-area-b4-01`) | Active |
| `cam-area-b3-01` | `occupancy-area-b3-01` | All Area `B3` bays | 1 API key (`occupancy-area-b3-01`) | Active rollout |
| `cam-area-b1-01` | `occupancy-area-b1-01` | All Area `B1` bays | 1 API key (`occupancy-area-b1-01`) | Planned |
| `cam-area-b5-01` | `occupancy-area-b5-01` | All Area `B5` bays | 1 API key (`occupancy-area-b5-01`) | Planned |

### Can the System Run Multiple Cameras Simultaneously?

**Yes.** The occupancy stack is designed to run multiple areas (`B2`, `B3`, `B4`, and future `B1`, `B5`) concurrently:

1. **Shared Model Sessions, Isolated Camera State:** In `scripts/lots_b2_b4/fusion_4model.py`, ONNX Runtime sessions (`YOLOX-L` and `MiDaS-small`) are cached once under `_ONNX_CACHE` (guarded by `_CACHE_LOCK`), so multiple camera threads/workers share model weights in memory instead of reloading models per lot. Meanwhile, each camera maintains its own homography, `PolygonMap`, `boot_id`, `FreshnessTracker`, `StateSmoother`, and monotonic `sequence_number`.
2. **Per-Device Independent Streams:** In deployment (`python -m parking_occupancy --config configs/cameras/area_b<n>.yaml`), each camera authenticates independently with its own `producer_id` and per-device API key so a stall or offline event on one camera never blocks or corrupts telemetry from other areas.
3. **Combined Local Throughput:** Running `scripts/run_lots_b2_b4_local.py` already executes full 4-model inference sequentially across both calibrated areas (`B2` + `B4`, 49 slots) in a single run, and lightweight change-monitoring across `B2, B3, B4` easily leaves headroom to run full 4-model inference well within a `<= 30s` cycle whenever active traffic demands it.

---

## 4. Reading Format (CSV-Driven HTTP `POST` Contract & Fields)

### CSV Output -> HTTP `POST` Mapping

Our 4-model pipeline (`scripts/lots_b2_b4/fusion_4model.py`) produces the authoritative per-slot CSV row schema:

```csv
area,slot_id,initial_state,model_1,model_2,model_3,model_4,mean,final_state,final_id
B2,L1,occupied,0.9119,0.5144,0.9466,0.8457,0.8261,occupied,1
B2,L3,available,0.0787,0.1186,0.0000,0.1052,0.0822,available,0
B2,R7,available,0.9180,0.7700,0.5360,0.8618,0.7913,occupied,1
B4,L1,occupied,0.9044,0.5600,0.7900,0.8930,0.8242,occupied,1
B4,MB1,occupied,0.7951,0.5600,0.7500,0.8930,0.7998,occupied,1
```

- **Weighted Fusion Formula:** `mean = 0.15 * model_1 (Geometry) + 0.15 * model_2 (Contextual) + 0.20 * model_3 (YOLOX-L) + 0.50 * model_4 (Top-Down BEV Re-Analysis)`
- **Final Classification:** `final_state = "occupied"` (`final_id = 1`) when `mean >= 0.50`; `final_state = "available"` (`final_id = 0`) when `mean < 0.50`.

### HTTP Transport & Envelope (`CANDIDATE-2026-10-30.2`, `schema_version: "1.0.0"`)

We **accept your proposed HTTP `POST` JSON contract** (`202 Accepted` Receipt):
- **`POST /api/v1/observations/occupancy`** — slot observations driven by the finalized CSV rows (or stale `unknown` updates)
- **`POST /api/v1/devices/heartbeats`** — periodic device liveness pings

Implemented in `src/parking_occupancy/events.py` (`build_occupancy_event` & `build_heartbeat_event`):

| Backend Question | Wire JSON Field | Type / Rule | Source / Semantics |
|---|---|---|---|
| Unique event ID per reading? | `event_id` | UUIDv4 string | **Yes.** Unique per emitted reading; **preserved identically on retries** (bounded retry queue `maxlen=100`, `2.0s–60.0s` backoff) for backend deduplication / idempotency. |
| Schema & event discriminator | `schema_version`, `event_type` | `"1.0.0"`, `"occupancy_observation"` \| `"device_heartbeat"` | Constant schema version and typed event names. |
| Timestamp | `occurred_at` | RFC 3339 UTC string (`...Z`) | UTC emission timestamp (`YYYY-MM-DDTHH:MM:SSZ`), preserved across retries. |
| Producer / key identity | `producer_id` | String | Camera `device_id` (e.g. `occupancy-area-b2-01`). |
| Correlation ID | `correlation_id` | `null` | **Always `null`** per decoupling rules (never coupled to gate/access attempts). |
| Lot & slot identity | `data.lot_code`, `data.space_id` | `"B"`, UUID string | Mapped from `(area, slot_id)` in `configs/polygons/area_b*.json`. |
| Status | `data.physical_status` | `"occupied"` \| `"available"` \| `"unknown"` | From CSV `final_state` (`final_id` `1 -> occupied`, `0 -> available`), or `unknown` on stale frame (`> 30.0s`). |
| Confidence | `data.confidence` | Float `[0.0, 1.0]` or `null` | Numeric score when `occupied`/`available`; **strictly `null`** when `unknown`. *(See calibration note below.)* |
| Process restart detection | `data.boot_id` | UUIDv4 string | Generated once per service process startup. |
| Sequence number | `data.sequence_number` | Monotonic integer (`>= 1`) | Incremented strictly per emitted observation within a `boot_id`. |
| Calibration & model metadata | `data.calibration_version`, `data.model_version` | Strings | e.g. `area-b2-4model-v1` and `sentry-4model-b2-b4-v1`. |

> **Important Clarifying Question for Backend on `confidence` when `physical_status == "available"`:**
> In the CSV, `mean` is the fused **occupancy probability** $P(\text{occupied}) \in [0, 1]$, so an empty slot like `B2 L3` has CSV `mean = 0.0822`. On the wire (`CANDIDATE-2026-10-30.2` / `06_SMOOTHING_FRESHNESS_AND_EVENTS.md`), `data.confidence` represents confidence *in the reported `physical_status`* (`1 - 0.0822 = 0.9178` certainty that `L3` is `"available"`). Please confirm that you expect status certainty (`0.9178` for `available`) rather than raw `P(occupied)` (`0.0822`). Both are reflected in [`sample_b2_b4_messages.json`](sample_b2_b4_messages.json).
>
> *Optional batch note:* For the high-change full final-state updates (`>= 2` full refreshes/minute across active lots), we can either stream the 22–27 slot `occupancy_observation` events per area to `POST /api/v1/observations/occupancy` or wrap them into a single JSON array if you expose a `/batch` endpoint.

---

## 5. Reporting Frequency, Lightweight Change-Monitor Script, & Freshness / Silent-Camera Rules

### Two-Tier Change-Monitor Cadence

To avoid flooding the backend (or wasting GPU/CPU cycles) on stationary overnight or static lots while ensuring rapid updates during busy arrival/departure windows, reporting uses a **two-tier monitor architecture**:

1. **Quiet / No-Significant-Change State (`occupancy` `POST`s suppressed):**
   - A **small, lightweight monitor script** watches incoming camera frames / bay ROIs continuously for visual change.
   - **When there is no significant change in vehicles**, no heavy 4-model CSV regeneration or occupancy `POST` spam is sent to the backend.
   - Only `device_heartbeat` events continue every **`60s`** per camera so the backend knows the camera is healthy. *(Note: in quiet periods, the baseline `EventEmitter` can optionally send an anti-entropy refresh every `refresh_interval_s = 300.0s` if desired, or keep occupancy traffic strictly change-driven.)*
2. **Active / High-Change State (`>= 2` full final-state updates per minute):**
   - When the monitor script detects significant vehicle changes across bays (entries, exits, or multi-slot visual deltas), it invokes full 4-model fusion (`M1–M4`), updates the authoritative CSV, and ships **updated full final-state data to the backend at least 2 times every 1 minute (`<= 30s` cadence)** for each active monitored camera (`B2`, `B4`, and `B3` when onboarded).
   - Once traffic settles and the CSV state stabilizes, the pipeline returns cleanly to quiet monitoring (`60s` heartbeats only).

### Silent-Camera Timeout -> `unknown` (Critical Heartbeat vs. Occupancy Rule)

| Layer | Threshold | Trigger Condition | Effect |
|---|---|---|---|
| **Camera-side (`FreshnessTracker`)** | **`30.0 seconds`** (`freshness_threshold_s: 30.0`) | No freshly processed video frame from camera source for `> 30.0s` | Transitions every space on that camera to `physical_status: "unknown"`, `confidence: null`, and emits observations. Heartbeats never reset frame freshness. |
| **Backend-side (Recommended)** | **`120–180 seconds`** (`2–3` missed `60s` heartbeats) | No **`device_heartbeat`** *and* no `occupancy_observation` received for `120–180s` | Marks all spaces under that camera's `producer_id` as `unknown` (`stale`) on the dashboard/API. |

> **Critical Backend Design Rule:** Because occupancy `POST`s are **intentionally suppressed when parked cars are stationary**, the backend **MUST use `device_heartbeat` (`60s` interval) — never occupancy message silence alone —** to determine whether a camera has gone offline. If a backend timeout waits only for `occupancy_observation` messages, a quiet lot with zero car movement would falsely flip to `unknown`.

---

## 6. Reported States

The service emits strictly **3 states** (`PhysicalStatus` in [`src/parking_occupancy/geometry.py`](../src/parking_occupancy/geometry.py)):

| `physical_status` | CSV `final_id` | `confidence` Rule | When Emitted |
|---|---|---|---|
| `"occupied"` | `1` | Float in `[0.50, 1.00]` | Vehicle confirmed inside stall polygon (`mean >= 0.50` and temporally smoothed). |
| `"available"` | `0` | Float in `[0.00, 1.00]` | Stall clear (`mean < 0.50` and temporally smoothed). |
| `"unknown"` | *(freshness override)* | **`null` (mandatory)** | Camera feed silent/stale `> 30.0s`, disconnected, occluded, or awaiting startup calibration. |

- **No `"uncertain"` enum state:** We never emit an `"uncertain"` string status; ambiguity is represented strictly by lowering the numeric `confidence` score (and borderline frame jitter is filtered by temporal hysteresis smoothing before emission).

---

## 7. Test Data, CSV Snapshots, & Local Simulator Commands

### 1. Sample Message File (`CANDIDATE-2026-10-30.2`)

Use **[`docs/sample_b2_b4_messages.json`](sample_b2_b4_messages.json)** as your JSON integration fixture. It is built directly from real 4-model audit results and includes:
- Verbatim excerpt of `outputs/lots_b2_b4/lot_b2_b4_occupancy.csv` rows
- Valid `occupancy_observation` payloads for:
  - `B2 L1` (`car_only`, `occupied`, CSV `mean = 0.8261`)
  - `B2 L3` (`car_only`, `available`, CSV `mean = 0.0822` -> wire `confidence = 0.9178`)
  - `B2 R7` (`car_only`, far-end stacked slot recovered from `initial_state=available` to `occupied`, CSV `mean = 0.7913`)
  - `B4 L1` (`car_only`, `occupied`, CSV `mean = 0.8242`)
  - `B4 MB1` (`motorcycle_only`, `occupied`, CSV `mean = 0.7998`)
  - `B2 L1` stale transition (`physical_status = "unknown"`, `confidence = null`)
- Valid `device_heartbeat` payloads for `occupancy-area-b2-01` and `occupancy-area-b4-01`

### 2. Local CSV & 4-Model Pipeline Runner (Produces 49-Bay CSV + Mock SENTRY Events)

From `ai_services/parking_occupancy/`:

```bash
python scripts/run_lots_b2_b4_local.py
```

Outputs `outputs/lots_b2_b4/lot_b2_b4_occupancy.csv` (49 rows across `B2` & `B4`: `48/49` occupied, `B2 = 21/22`, `B4 = 27/27`), per-lot and combined 2D slot maps (`lot_b2_2d_map.png`, `lot_b4_2d_map.png`, `lot_b2_b4_combined_2d_map.png`), and validates 49 mock SENTRY events (`correlation_id: null`). *(Note: `outputs/` is gitignored locally; run the command above to generate the full 49-row CSV.)*

### 3. Single-Camera CLI Simulator (`SyntheticDetector` Mock Mode)

Run without external YOLO downloads or a live backend server (with `PYTHONPATH=src`, or after `pip install -e .`):

```bash
PYTHONPATH=src python -m parking_occupancy \
  --config configs/cameras/area_b2.yaml \
  --still ui/samples/area_b2_raw.jpg \
  --synthetic-detector \
  --backend-mode mock
```

Swap `--config configs/cameras/area_b4.yaml` (with a B4 image) to simulate Area `B4`. Switching to `--backend-mode real` only requires your staging ingest URL and per-device API keys (`SEC-01`).

### 4. Wire Contract & Idempotency Tests

```bash
PYTHONPATH=src python -m pytest tests/contract/test_events.py -q
```

---

## Backend Team Action Items to Unblock Integration

1. **Canonical Slot Mapping Seed:** Provide your `(area, slot_id) -> (B-P... display code, database UUID)` table for `B2` (22 bays), `B4` (27 bays), and upcoming `B3`.
2. **Per-Device Credentials (`SEC-01`):** Issue one API key per `device_id`: `occupancy-area-b2-01`, `occupancy-area-b4-01`, and `occupancy-area-b3-01` (with `b1-01` / `b5-01` reserved for expansion).
3. **`confidence` Convention for `available` Slots:** Confirm whether you expect certainty in `physical_status` (`1 - P(occupied) = 0.9178` for `B2 L3`) or raw `P(occupied)` (`0.0822`).
4. **Silent-Camera Timeout Confirmation:** Confirm `120–180s` timeout driven by **`device_heartbeat`** (`60s` interval) rather than `occupancy_observation` silence so quiet unchanged lots do not false-stale.
5. **High-Change Refresh Ingestion & Staging URL:** Confirm whether `<= 30s` full final-state updates (`>= 2` refreshes/min during active changes) should hit single-event `POST /api/v1/observations/occupancy` or an optional batch endpoint, and share the staging ingest base URL.
