# Supporting Signal — Lightweight Object Detection (Lower Weight & Confidence Band)

> **Status:** Adapter implemented in [`inference.py`](../src/parking_occupancy/inference.py) (`YOLODetector`, `SyntheticDetector`) | **Extensions planned:** Lower proposal gate tuning, uncertainty band export, and ground-footprint tagging  
> **Data Policy:** Uses `SyntheticDetector` with dummy boxes in CI/dev; optional CPU `yolov8n.pt` weights (gitignored)

---

## 1. Architectural Role: Supporting Proposal, NOT Primary Judge

In this architecture, AI object detection (e.g., `YOLOv8n`) is deliberately **not** treated as the main or most accurate authority for slot occupancy:

- **What it DOES:** Scans the unwarped raw frame (CCTV or optimized angle) to propose candidate bounding boxes where vehicles (`car`, `motorcycle`) might exist.
- **What it DOES NOT do:** It never makes the final `occupied / available` call on its own, and its raw confidence score is **never** treated directly as slot occupancy confidence.

### Why We Deliberately Assign Lower Trust / Confidence to Object Detection

1. **Oblique CCTV Distortion & Partial Occlusion:** In angled CCTV footage (e.g., far rows or behind vans/trees), vehicles overlap heavily. Standard object detectors frequently miss partially occluded cars or report low scores (`0.20–0.45`).
2. **Threshold Tradeoff:** If we used a high standalone YOLO confidence threshold (e.g. `0.70+`), partially occluded vehicles would vanish completely and falsely make slots look empty. Therefore, we run YOLO with a **lower acceptance floor** (e.g., `0.20–0.35`) so candidates are surfaced, **while assigning YOLO a lower weight and wider/lower confidence interval** inside the downstream **Cognitive Fusion AI Model**.
3. **Pedestrian & Clutter Immunity:** YOLO detects `person` (COCO class `0`) or background traffic outside polygons, which must never falsely occupy a bay.

---

## 2. Detection Pipeline & Class Filtering

In [`src/parking_occupancy/inference.py`](../src/parking_occupancy/inference.py):

| Parameter / Symbol | Current Implementation | Target Supporting Configuration |
|---|---|---|
| **Input Frame** | Original unwarped BGR frame (`Frame.image`) | Same — always detect before perspective warp to avoid blur |
| **Runtime Device** | CPU (`device="cpu"`) | CPU-first (lightweight `yolov8n.pt` or synthetic test double) |
| **Target Classes** | `COCO_CLASS_CAR = 2`, `COCO_CLASS_MOTORCYCLE = 3` (`VEHICLE_CLASSES`) | Keep `car` & `motorcycle` strict; optionally track `bus=5` / `truck=7` as large-vehicle spillover hazards |
| **Confidence Threshold** | Default `0.4` in `YOLODetector.__init__` | Lower configurable gate (e.g., `0.25`) for `cctv_oblique` so occluded cars still pass to Contextual Awareness |
| **NMS IoU Threshold** | Default `0.45` | Tuned (`0.50–0.60`) so tightly packed adjacent cars in oblique CCTV views do not suppress one another |
| **Test Double** | `SyntheticDetector(detections)` | Primary source for deterministic dummy-data scenario testing |

### Detection Output Record

Each candidate box produces a `Detection`:
```python
@dataclass(frozen=True)
class Detection:
    vehicle_class: VehicleClass  # "car" | "motorcycle"
    confidence: float            # Raw detector score [0.0, 1.0]
    x1: float
    y1: float
    x2: float
    y2: float
```
From `(x1, y1, x2, y2)`, Geometry extracts both the centroid `(cx, cy)` and the planned **ground-plane footprint quadrilateral** for projection into BEV space.

---

## 3. Mapping Detector Confidence into a Lower Confidence Band

Because object detection is only a supporting sensor (not the main model), its raw score $c_{\text{yolo}}$ is **compressed and bounded** before entering the Cognitive Fusion AI Model:

1. **Supporting Weight Cap ($w_{\text{yolo}}$):**
   - In the Cognitive Fusion AI Model, raw YOLO score contributes a **minority weight** (e.g., $w_{\text{yolo}} = 0.25\text{–}0.35$), whereas geometric slot area overlap, remaining-area feasibility, contextual occlusion state, and the **1-minute temporal mean** carry the decisive weight.
2. **Lower / Wider Confidence Interval Assignment:**
   - Rather than passing `0.85` as an absolute certainty, a standalone raw detection carries a wide uncertainty band:
     - **Single-frame YOLO-only confidence cap:** e.g. $c_{\text{det\_bounded}} = \min(c_{\text{yolo}}, 0.75)$
     - **Uncertainty margin ($\pm \delta_{\text{det}}$):** wider for `cctv_oblique` (e.g. $\delta = 0.20\text{–}0.30$) than for `optimized_angle` (e.g. $\delta = 0.10\text{–}0.15$).
   - Even if YOLO outputs `0.0` (no box found), the fusion model does **not** jump to `available` with `1.0` confidence when Contextual Awareness reports that the slot is occluded or partially consumed by a neighbor!

---

## 4. Safety & Failure Rules

1. **Missing Weights (`MissingWeightsError`)**:
   - In real-detector mode, missing weights raise `MissingWeightsError` rather than silently returning `[]` detections (which would falsely look like an empty parking lot).
   - In dev/CLI mode without weights, pass `--synthetic-detector` or leave `model_path` empty so `SyntheticDetector` runs explicitly.
2. **Model Manifest Governance**:
   - Record checksum, license (`AGPL-3.0` for YOLOv8n), input size, and thresholds in `models/manifest.example.json`. Never commit `.pt` / `.onnx` binary weights to Git.
3. **Zero Identity Leakage**:
   - Detector outputs vehicle class + box coordinates only—never license plates, faces, or user UUIDs.

---

## 5. Area B2 Dual-Pass YOLOX-L Tiled + Far-End Zoomed Crop Strategy (`scripts/area_b2/parking_ai_check.py`)

In the Area B2 audit ([`parking_ai_check.py`](../scripts/area_b2/parking_ai_check.py) & [`topview_slots.py`](../scripts/area_b2/topview_slots.py)), single-frame whole-image inference misses small vehicles at the distant edge of the `2048×1536` frame. To counter oblique scale loss using pretrained `yolox_l.onnx` (COCO vehicle classes `2=car, 3=motorcycle, 5=bus, 7=truck`):

1. **Tiled Full-Image Pass (`yolox_vehicles`, gate `0.30`):**
   - Runs full-frame inference plus 5 overlapping tiles (`(0,300,1100,1236)`, `(948,300,2048,1236)`, `(0,500,1100,1536)`, `(948,500,2048,1536)`, `(500,400,1550,1300)`) with tile-edge box clipping and OpenCV NMS (`iou=0.35`).
2. **Zoomed $640 \times 640$ Far-End Passes (`FAR_ZOOMS`, gate `0.25`):**
   - `far-left`: `(640, 430, 1010, 640)` — confirmed `L1–L10` present and verified **no 11th car** in `L11`.
   - `far-right`: `(1230, 430, 1570, 640)` — surfaced ~2 dark vehicles stacked behind the silver van (`R9–R10`, marked `probable`).
3. **Storage & Download Policy:**
   - Binary weights (~280 MB for `yolox_l.onnx` + `midas_small.onnx`) are excluded from Git and downloaded on demand into `models/` on first script execution.

