import cv2
from ultralytics import YOLO

# Load YOLO26s pretrained on COCO (80 object classes, NMS-free end-to-end)
model = YOLO("yolo26s.pt")

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break

    # Run detection
    results = model(frame, verbose=False)

    # Draw annotated frame
    annotated = results[0].plot()

    cv2.imshow("YOLO26s Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
