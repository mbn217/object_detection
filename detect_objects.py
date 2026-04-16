import argparse
import time
import cv2
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="YOLO live object detection")
parser.add_argument("--model", default="yolo26s.onnx", help="Path to model file (e.g. yolo26s.pt, yolo26s.onnx)")
args = parser.parse_args()

model = YOLO(args.model)
print(f"Loaded model: {args.model}")

# Open default camera using DirectShow backend (Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Cannot open camera. Trying alternative indices...")
    for i in range(1, 5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"Opened camera at index {i}")
            break
    if not cap.isOpened():
        print("Error: No camera found")
        exit(1)

print("Press 'q' to quit")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # Run detection
    results = model(frame, verbose=False)

    # Draw annotated frame
    annotated = results[0].plot()

    # Calculate and overlay FPS
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("YOLO26s Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
