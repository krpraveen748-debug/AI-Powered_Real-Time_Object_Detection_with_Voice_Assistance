"""
============================================================
  WEBCAM OBJECT DETECTION — YOLOv8 (Ultralytics)
============================================================

Simple live webcam object detection. Just hit Run ▶️.

INSTALL FIRST (run once in terminal):
    pip install ultralytics opencv-python

Press 'q' on the webcam window to quit.
============================================================
"""

import cv2
from ultralytics import YOLO

# ============================================================
# ⚙️  CONFIG
# ============================================================
MODEL_SIZE = "n"        # n = fastest, x = most accurate
CONF_THRESHOLD = 0.25  # lower = detects more, but more false positives

# Only detect these objects (COCO class IDs). Set to None to detect all 80 classes.
#   0  = person
#   24 = backpack   (bag)
#   26 = handbag    (bag)
#   28 = suitcase
#   67 = cell phone
#   77 = teddy bear
TARGET_CLASSES = [0, 24, 26, 28, 67, 77]
# ============================================================


def main():
    print("[INFO] Loading model...")
    model = YOLO(f"yolov8{MODEL_SIZE}.pt")
    print("[INFO] Model loaded. Starting webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not access the webcam. Check it's connected and not in use by another app."
        )

    print("[INFO] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        results = model(
            frame,
            conf=CONF_THRESHOLD,
            classes=TARGET_CLASSES,
            verbose=False
        )
        annotated_frame = results[0].plot()

        num_objects = len(results[0].boxes)
        cv2.putText(
            annotated_frame,
            f"Objects detected: {num_objects}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Object Detection - Webcam", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    for _ in range(4):
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
