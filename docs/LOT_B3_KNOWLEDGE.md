# Area B3 — Layout, Slot Knowledge & Monitoring Guide (PROVISIONAL)

Source: one hand-held photo (`ui/samples/area_b3_raw.jpg`, 2576×1932). Assumed to be **B3** (the Q&A doc lists B3 as "awaiting its fixed camera reference frame"). It is **not** the fixed camera frame, so every polygon is a first draft and there is no homography yet.

## 1. What the lot looks like

A small open lot in front of a white/brick building, seen from an upper floor, with strong building shadow across the lower third of the frame.

| Zone | Slots | Geometry in the photo |
|---|---|---|
| Left row `L1–L9` | 9 | Angled bays along the covered walkway, cars nose to the lower right. Numbered far (top of photo) → near, like B2/B4. Painted dividers are faint, so bay edges are inferred from the cars. |
| Right row `R1–R9` | 9 | Side-on cars along the kerb. `R1–R6` are squeezed into very few pixels by perspective, so they are the hardest to monitor. |
| `R10` | 1 | **Accessible (PWD) bay**, painted symbol, clearly marked lines, empty in the photo. |
| Centre `C1` | 1 | Painted bay holding the white Camry. Low geometry confidence. |
| Informal (not bays) | `A1` jeepney, `A2` black sedan, `A3` black sedan (top) | Parked in the aisle. Tracked in `informal_zones`, **never counted** as bay occupancy. |

Total: **20 car bays** (`car_only`; B3 has no motorcycle area, and `R10` is the last right-row bay, there is no `R11`). Provisional `space_id`s are `30000000-0000-4000-8000-000000000301`…`0320` (B2 uses 101–122, B4 uses 201–227), in the order L1–L9, R1–R10, C1. `L9` is level with `R10` in the plan.

## 2. Baseline state in the photo (manual reading)

`19 occupied · 1 available (R10)` → 19/20 bays occupied (95%). All of `L1–L9`, `R1–R9` and `C1` hold a vehicle. This is a human reading, written to `outputs/lot_b3/lot_b3_occupancy.csv` in the fusion-CSV schema with `model_1…model_4` and `mean` left blank, because no model was run. Do not treat it as model output.

## 3. What will trip the monitor, and how it is handled

| Hazard | Why it matters here | Handling |
|---|---|---|
| Moving building shadow / exposure | The shadow edge crosses `C1`, `R8–R10` and `L9`. | Change score is normalised cross-correlation inside each polygon, so brightness shifts barely move it (tested: a 30% darkening triggers nothing). |
| Pedestrians and umbrellas | Two people with umbrellas cross `C1` in this very photo. | A change must persist 3 consecutive polls (~6 s). A one-frame blip does not trigger (tested). `person`/`umbrella` are listed under `ignore`. |
| Aisle vehicles (jeepney, sedans) | They move often and sit between rows. | Kept as `informal_zones`; they never change a bay's state. |
| Compressed `R1–R6` | Few pixels per bay, cars overlap in projection. | Expect lower confidence; check these first when tuning on the real camera. |
| Accessible bay `R10` | Physical occupancy only. | Reservation / PWD flag belongs in backend space metadata (SEC-02). |

## 4. Monitoring behaviour (`scripts/lot_b3/change_monitor.py`)

- **QUIET:** no occupancy POSTs; `device_heartbeat` every 60 s.
- **ACTIVE** (a bay changed and stayed changed): full 4-model update of all slots immediately, then every 20 s (≤ 30 s, so ≥ 2 per minute) until 2 consecutive cycles show no change.
- **STALE** (no frame for > 30 s): every slot is emitted `unknown`, `confidence: null`; heartbeats continue.
- Self-test (`python scripts/lot_b3/change_monitor.py --selftest`) passes with simulated frames from the B3 photo: 5 heartbeats in 5 quiet minutes, no trigger on a shadow shift or a pedestrian blip, a departing L5 car triggers 3 full updates 20 s apart then settles, and a silent camera flips all 20 slots to `unknown`.

Caveats: the inference hook is a stub returning the manual baseline; replace it with `fusion_4model.py`. The event builder is a fallback copy of the documented envelope; swap in `build_occupancy_event` / `build_heartbeat_event` from `src/parking_occupancy/events.py`. The self-test uses synthetic edits to one still photo, not real video.

## 4b. Top-down view and slot plan (`scripts/lot_b3/topdown_b3.py`)

- `outputs/lot_b3/lot_b3_topview.png` is the photo warped to a ground-plane view (60 px/m, 5 m scale bar, 5 m grid) with the bay rectangles drawn on it; `lot_b3_topview_raw.png` is the same without overlay; `lot_b3_slot_plan.png` is the schematic.
- **How the homography was made:** the painted accessible bay `R10` is a rectangle 2.4 m wide, so its four painted corners fix the plane. Its depth (4.33 m) is not painted, so it was solved so the red Wigo's wheelbase (2.45 m) comes out right.
- **Checks that passed:** the right-row bays land 2.4 m apart; the Wigo's centre falls in the bay above `R10`; the left row's painted dividers come out parallel and ~2.3 m apart, and 9 of them fit `L1–L9` (the plan then shifts the left row ~1.2 m so `L9` is level with `R10`, as confirmed by you).
- **Limits:** ground only. Car roofs are above the plane, so in the warp cars smear away from the camera and far cars (`R1–R4`, `L1–L3`) sit up to about one bay off their ground bay. Use `plan_polygon_m` / `ground_polygon_px` (pavement) for geometry and `polygon_px` (covers the car body) for change-monitoring. Accuracy is roughly ±0.5 m near `R8–R10`, worse toward the top.
- Run order: `build_b3.py` first, then `topdown_b3.py` (the second adds the plan fields to the JSON; re-running `build_b3.py` alone drops them).
- **Open finding:** in the warp, the jeepney `A1`, black sedan `A2` and Camry `C1` line up in one column beside a long painted line, so the centre may be a column of parallel bays (`C1–C4`) rather than aisle parking. Only `C1` has painted lines I can see; confirm on site.

## 5. To finish B3 calibration

1. Capture a frame from the **fixed** camera and re-draw the polygons (edit the box lists in `scripts/lot_b3/build_b3.py`, run it, check `lot_b3_overlay.png`).
2. Re-pick ground control points on the fixed camera frame and re-run `topdown_b3.py` (a provisional homography from the photo is already in `area_b3.homography.json`).
3. Confirm the centre bay and the aisle (is `C1` a real bay row or one painted slot?).
4. Confirm that `R10` is the last bay in the right row.
5. Run the 4-model fusion and compare with the manual baseline above.
6. Get the backend `(area, slot_id) → (space_id, B-P… code)` seed and the API key for `occupancy-area-b3-01`.

## 6. Discrepancy to resolve in B4

The B4 slot-plan images show **10** motorcycle bays (`MB10` sits apart from `MB1–MB9`, partly out of view) and **4 informal aisle vehicles** (`A1–A4`, total 32 vehicles), while the Q&A doc and `lot2_occupancy.csv` list **27** B4 bays (`MB1–MB9` only). Decide whether `MB10` becomes a 28th bay, and whether the aisle vehicles stay excluded. Also note the zip's `slot_occupancy.py` imports `topview_slots.py` and `parking_lot2.py`, which were not in the zip, so it cannot run as supplied; I did not reuse it.
