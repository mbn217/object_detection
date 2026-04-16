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
├── detect_objects.py           # Basic detection (PyTorch, single-threaded)
├── detect_objects_openvino.py  # OpenVINO FP16 optimized detection
├── detect_objects_threaded.py  # 3-thread pipeline with batch detection + BYTETracking
├── requirements.txt            # Python dependencies
└── README.md                   # This file
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

## OpenVINO Optimized Detection

[detect_objects_openvino.py](detect_objects_openvino.py) exports the model to OpenVINO FP16 format and benchmarks it against the standard PyTorch model before starting live detection.

### Install the OpenVINO dependency

```bash
pip install openvino
```

### Run

```bash
python detect_objects_openvino.py
```

On first run, the script will:
1. Export `yolo26s.pt` → `yolo26s_openvino_model/` (FP16, done once)
2. Run a **30-frame benchmark** on both models and print a speedup summary to the console
3. Open the live camera window with an **FPS overlay** and the recorded speedup displayed on-screen

Expected speedup on Intel hardware: **up to 3x faster** CPU inference vs PyTorch.

> OpenVINO is optimized for Intel CPUs, integrated GPUs, and NPUs. Results will vary on non-Intel hardware.

## Multithreaded Detection + Tracking Pipeline

[detect_objects_threaded.py](detect_objects_threaded.py) splits the pipeline across three threads connected by shared queues:

```
[Camera Thread] ──► Frame Buffer ──► [Detection Thread] ──► Detection Buffer ──► [Tracking Thread] ──► Display Buffer ──► [Main]
```

| Thread | Work | Buffer Out |
|---|---|---|
| Camera | `cap.read()` at full camera rate | Frame Buffer |
| Detection | Batch `BATCH_SIZE` frames → single forward pass | Detection Buffer |
| Tracking | BYTETrack → assign persistent track IDs + draw | Display Buffer |
| Main | `cv2.imshow` + per-stage FPS overlay | — |

### Why threads and not processes?

PyTorch and OpenCV release Python's GIL during their C-level operations, so threads achieve real CPU parallelism for the heavy compute. Multiprocessing would add frame-serialization overhead that outweighs any benefit.

### Run

```bash
python detect_objects_threaded.py
```

The live window shows a bottom panel with FPS for each pipeline stage so you can see exactly where the bottleneck is.

Tune `BATCH_SIZE` at the top of the file (default `4`) — larger batches improve GPU/CPU utilisation but add latency.

## References

- [Ultralytics YOLO26 Docs](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics OpenVINO Integration](https://docs.ultralytics.com/integrations/openvino/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
