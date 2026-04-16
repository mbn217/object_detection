# Real-Time Object Detection with YOLO26

Live object detection from a webcam using [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) — an NMS-free, end-to-end model pretrained on COCO (80 classes).

## Requirements

- Python 3.8+
- A webcam connected to your machine

## Installation

1. Clone or download this repository.

2. Create and activate a virtual environment:

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Recommended) Make sure you have the latest version of `ultralytics` to support YOLO26:

```bash
pip install -U ultralytics
```

> The `yolo26x.pt` model weights will be downloaded automatically on first run.

> To deactivate the virtual environment when done, run `deactivate`.

## Running the Detection Script

```bash
python detect_objects.py
```

- A window will open showing the live camera feed with bounding boxes drawn around detected objects.
- Press **`q`** to quit.

## File Structure

```
object_detection/
├── detect_objects.py   # Main detection script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How It Works

1. Loads the `yolo26x` model pretrained on the COCO dataset (80 object classes).
2. Opens the default webcam using the DirectShow backend (Windows). If the default index fails, it automatically tries indices 1–4.
3. Runs YOLO26 inference on each captured frame.
4. Draws annotated bounding boxes and labels on the frame using `results[0].plot()`.
5. Displays the annotated feed in a window until `q` is pressed.

## Model Variants

You can swap `yolo26x.pt` in `detect_objects.py` for a lighter model if needed:

| Model      | Size  | mAP  | Speed (CPU) |
|------------|-------|------|-------------|
| yolo26n.pt | Nano  | 40.9 | Fastest     |
| yolo26s.pt | Small | 48.6 | Fast        |
| yolo26m.pt | Med   | 53.1 | Balanced    |
| yolo26l.pt | Large | 55.0 | Accurate    |
| yolo26x.pt | XL    | 57.5 | Most accurate |

## References

- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
