import argparse
import time
import os
import numpy as np
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLO OpenVINO optimized detection")
parser.add_argument("--model", default="yolo26s.pt", help="Path to PyTorch model file (e.g. yolo26s.pt)")
args = parser.parse_args()

PT_MODEL = args.model
MODEL_STEM = os.path.splitext(os.path.basename(PT_MODEL))[0]
OV_MODEL_DIR = f"{MODEL_STEM}_openvino_model/"
WARMUP_FRAMES = 30  # frames used for benchmark comparison


def benchmark(model, dummy_frame, n=WARMUP_FRAMES, label=""):
    """Run n inference passes and return average FPS."""
    # warm-up pass
    model(dummy_frame, verbose=False)
    start = time.time()
    for _ in range(n):
        model(dummy_frame, verbose=False)
    elapsed = time.time() - start
    fps = n / elapsed
    print(f"  {label:<20} {fps:>7.1f} FPS  ({elapsed:.2f}s for {n} frames)")
    return fps


# ---------------------------------------------------------------------------
# 1. Export to OpenVINO (FP16) if not already done
# ---------------------------------------------------------------------------
if not os.path.isdir(OV_MODEL_DIR):
    print(f"[INFO] Exporting {PT_MODEL} to OpenVINO (FP16)...")
    pt_model = YOLO(PT_MODEL)
    pt_model.export(format="openvino", half=True)
    print(f"[INFO] Export complete → {OV_MODEL_DIR}")
else:
    print(f"[INFO] OpenVINO model already exists at '{OV_MODEL_DIR}', skipping export.")

# ---------------------------------------------------------------------------
# 2. Load both models
# ---------------------------------------------------------------------------
print("\n[INFO] Loading models...")
pt_model = YOLO(PT_MODEL)
ov_model = YOLO(OV_MODEL_DIR)

# ---------------------------------------------------------------------------
# 3. Benchmark on a dummy frame
# ---------------------------------------------------------------------------
dummy = np.zeros((480, 640, 3), dtype=np.uint8)

print(f"\n[BENCHMARK] Warming up both models on a {dummy.shape[1]}x{dummy.shape[0]} dummy frame ({WARMUP_FRAMES} passes each):")
pt_fps = benchmark(pt_model, dummy, label="PyTorch (CPU)")
ov_fps = benchmark(ov_model, dummy, label="OpenVINO (FP16)")
speedup = ov_fps / pt_fps if pt_fps > 0 else 0
print(f"\n  Speedup: {speedup:.2f}x  ({'faster' if speedup >= 1 else 'slower'} with OpenVINO)\n")

# ---------------------------------------------------------------------------
# 4. Live detection using OpenVINO model
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("[INFO] Trying alternative camera indices...")
    for i in range(1, 5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"[INFO] Opened camera at index {i}")
            break
    if not cap.isOpened():
        print("[ERROR] No camera found")
        exit(1)

print("[INFO] Live detection started. Press 'q' to quit.\n")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot read frame")
        break

    results = ov_model(frame, verbose=False)
    annotated = results[0].plot()

    # FPS overlay
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(annotated, f"OpenVINO FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(annotated, f"Benchmark speedup: {speedup:.2f}x vs PyTorch", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

    cv2.imshow(f"YOLO26s — OpenVINO FP16", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
