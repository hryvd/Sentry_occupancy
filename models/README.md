# Models directory

Place model weight files here (not tracked in Git — see `.gitignore`).

## Required weights

| File | Source | License | Download |
|------|--------|---------|----------|
| `yolov8n.pt` | Ultralytics YOLOv8 | AGPL-3.0 | [ultralytics/assets](https://github.com/ultralytics/assets/releases) |

## Setup

```powershell
# From the parking_occupancy module root
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
# This auto-downloads yolov8n.pt to the current directory on first run
# Then move it here: models/yolov8n.pt
```

Then set `MODEL_PATH=models/yolov8n.pt` in your `.env`.

## Manifest

See `manifest.example.json` for the required manifest schema.
Every model version must have a matching manifest recording:
- `sha256_checksum` of the weights file
- `source_url` and `license`
- `input_resolution`, `confidence_threshold`, `iou_threshold`
- `evaluation_evidence` link

Large weight files (`.pt`, `.onnx`, `.bin`) are gitignored.
Only `manifest.example.json` and this README are tracked.

