"""
detect_objects_threaded.py

3-stage multithreaded detection pipeline matching the architecture:

  [Camera Thread]         → frame_buffer     (Frame Buffer)
  [Detection Thread]      → detection_buffer (Detection Buffer)  ← BATCH inference
  [Tracking Thread]       → display_buffer   (Tracking Buffer)
  [Main / Display Thread] → cv2.imshow

Why threading (not multiprocessing)?
  PyTorch/OpenCV release the GIL during C-level operations, so threads run
  truly in parallel for the heavy work. Multiprocessing would add
  frame-serialization overhead that outweighs any benefit here.

Performance gains:
  - Camera capture never blocks inference (separate thread + queue)
  - BATCH_SIZE frames are inferred in a single forward pass (GPU/CPU efficiency)
  - Tracking runs concurrently with next detection batch
"""

import time
import queue
import threading
from types import SimpleNamespace

import cv2
import numpy as np
from ultralytics import YOLO

# ── Try to import BYTETracker (internal ultralytics API) ─────────────────────
try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    _TRACKER_AVAILABLE = True
except ImportError:
    _TRACKER_AVAILABLE = False
    print("[WARN] BYTETracker not found – tracking thread will plot detections only.")

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "yolo26s"
PT_MODEL      = f"{MODEL_NAME}.pt"
BATCH_SIZE    = 32      # frames processed per model forward pass
FRAME_Q_MAX   = 16     # Frame Buffer   – drop oldest if camera outruns detection
DETECT_Q_MAX  = 8      # Detection Buffer
DISPLAY_Q_MAX = 4      # Tracking Buffer (display-ready frames)

TRACKER_CFG = SimpleNamespace(
    tracker_type      = "bytetrack",
    track_high_thresh = 0.5,
    track_low_thresh  = 0.1,
    new_track_thresh  = 0.6,
    track_buffer      = 30,
    match_thresh      = 0.8,
    fuse_score        = True,
)

# ── Shared state ──────────────────────────────────────────────────────────────
stop_event       = threading.Event()
frame_buffer     = queue.Queue(maxsize=FRAME_Q_MAX)
detection_buffer = queue.Queue(maxsize=DETECT_Q_MAX)
display_buffer   = queue.Queue(maxsize=DISPLAY_Q_MAX)

_fps_data = {"Camera": 0.0, "Detection": 0.0, "Tracking": 0.0, "Display": 0.0}
_fps_lock = threading.Lock()

def _record_fps(stage: str, count: int, elapsed: float) -> None:
    with _fps_lock:
        _fps_data[stage] = count / elapsed if elapsed > 0 else 0.0


# ── Thread 1: Camera ──────────────────────────────────────────────────────────
def camera_worker(cap: cv2.VideoCapture) -> None:
    """Capture frames at full camera rate, push to frame_buffer. Drop if full."""
    count, t0 = 0, time.time()
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        try:
            frame_buffer.put_nowait(frame)
        except queue.Full:
            # Buffer full – discard this frame so camera never blocks
            pass
        count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            _record_fps("Camera", count, elapsed)
            count, t0 = 0, time.time()


# ── Thread 2: Batch Detection ─────────────────────────────────────────────────
def detection_worker(model: YOLO) -> None:
    """
    Collect up to BATCH_SIZE frames from frame_buffer, run a SINGLE batched
    model forward pass, then push (frame, result) pairs to detection_buffer.
    """
    count, t0 = 0, time.time()
    while not stop_event.is_set():
        batch: list = []
        # Wait up to 50 ms to fill the batch; process partial batches too
        deadline = time.time() + 0.05
        while len(batch) < BATCH_SIZE and time.time() < deadline:
            try:
                batch.append(frame_buffer.get(timeout=0.01))
            except queue.Empty:
                pass

        if not batch:
            continue

        # One forward pass handles all frames in batch
        results = model(batch, verbose=False)

        for frame, result in zip(batch, results):
            try:
                detection_buffer.put((frame, result), timeout=0.05)
            except queue.Full:
                pass  # skip if tracking thread is falling behind

        count += len(batch)
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            _record_fps("Detection", count, elapsed)
            count, t0 = 0, time.time()


# ── Thread 3: Tracking ────────────────────────────────────────────────────────
def tracking_worker() -> None:
    """
    Read (frame, detections) from detection_buffer, update BYTETracker to
    assign persistent track IDs, draw annotated frame, push to display_buffer.
    """
    tracker = BYTETracker(TRACKER_CFG, frame_rate=30) if _TRACKER_AVAILABLE else None
    count, t0 = 0, time.time()

    while not stop_event.is_set():
        try:
            frame, result = detection_buffer.get(timeout=0.1)
        except queue.Empty:
            continue

        annotated = frame.copy()

        if tracker is not None:
            try:
                tracks = tracker.update(result, frame)
                for track in tracks:
                    x1, y1, x2, y2 = track.tlbr.astype(int)
                    tid  = int(track.track_id)
                    cls  = int(track.cls)   if hasattr(track, "cls")   else 0
                    conf = float(track.score) if hasattr(track, "score") else 0.0
                    name = result.names.get(cls, str(cls))
                    color = _track_color(tid)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        annotated, f"#{tid} {name} {conf:.2f}",
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                    )
            except Exception:
                # Fallback: plot raw detections without track IDs
                annotated = result.plot()
        else:
            # No tracker – just plot YOLO detections
            annotated = result.plot()

        try:
            display_buffer.put(annotated, timeout=0.05)
        except queue.Full:
            pass

        count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            _record_fps("Tracking", count, elapsed)
            count, t0 = 0, time.time()


def _track_color(track_id: int) -> tuple:
    """Return a consistent BGR colour for a given track ID."""
    np.random.seed(track_id * 7 + 13)
    return tuple(int(x) for x in np.random.randint(80, 230, 3))


# ── Main / Display ────────────────────────────────────────────────────────────
def main() -> None:
    print(f"[INFO] Loading {PT_MODEL} ...")
    model = YOLO(PT_MODEL)

    # Open camera
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
        return

    # Launch pipeline threads
    threads = [
        threading.Thread(target=camera_worker,    args=(cap,),   daemon=True, name="Camera"),
        threading.Thread(target=detection_worker, args=(model,), daemon=True, name="Detection"),
        threading.Thread(target=tracking_worker,                 daemon=True, name="Tracking"),
    ]
    for t in threads:
        t.start()

    print(f"[INFO] Pipeline started  |  batch={BATCH_SIZE}  |  Press 'q' to quit")
    print(f"[INFO] Tracker: {'BYTETrack' if _TRACKER_AVAILABLE else 'disabled (fallback to detection only)'}")

    count, t0 = 0, time.time()
    window_title = f"YOLO26s — Threaded Pipeline (batch={BATCH_SIZE})"

    while not stop_event.is_set():
        try:
            annotated = display_buffer.get(timeout=0.1)
        except queue.Empty:
            continue

        # Measure display FPS
        count += 1
        elapsed = time.time() - t0
        if elapsed >= 1.0:
            _record_fps("Display", count, elapsed)
            count, t0 = 0, time.time()

        # Overlay per-stage FPS
        with _fps_lock:
            stats = dict(_fps_data)

        stage_colors = {
            "Camera":    (0,   220,   0),
            "Detection": (0,   200, 255),
            "Tracking":  (255, 180,   0),
            "Display":   (255, 255, 255),
        }
        h = annotated.shape[0]
        panel_top = h - (len(stats) * 24 + 36)
        cv2.rectangle(annotated, (0, panel_top), (270, h), (0, 0, 0), -1)
        cv2.putText(annotated, f"Batch size: {BATCH_SIZE}",
                    (8, panel_top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
        for i, (stage, fps) in enumerate(stats.items()):
            cv2.putText(
                annotated,
                f"{stage:<10}: {fps:>5.1f} fps",
                (8, panel_top + 38 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                stage_colors[stage], 2, cv2.LINE_AA,
            )

        cv2.imshow(window_title, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    # Shutdown
    for t in threads:
        t.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()
