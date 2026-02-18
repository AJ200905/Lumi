# ===================== IMPORTS =====================
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ===================== LOAD YOLO MODEL =====================
model = YOLO("yolov8n.pt")  # lightweight model

# ===================== START WEBCAM =====================
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

print("🔴 Live detection started — press 'q' to quit")

# ===================== CONTINUOUS DETECTION LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO detection
    results = model(frame_rgb, verbose=False)

    h, w, _ = frame.shape
    action = "NO HAND"

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Using 'person' as hand proxy
            if label == "person" and conf > 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (x_center, y_center), 5, (255, 0, 0), -1)

                # Lamp movement logic
                if x_center < w // 3:
                    action = "MOVE LEFT"
                elif x_center > 2 * w // 3:
                    action = "MOVE RIGHT"
                else:
                    action = "CENTERED"

    # Display action
    cv2.putText(
        frame,
        action,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        3
    )

    cv2.imshow("Robotic Lamp - Live Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===================== CLEANUP =====================
cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped")
