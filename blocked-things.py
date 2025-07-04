from ultralytics import YOLO
import cv2
import time
import os
import requests

MODEL_PATH = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    print("📦 Downloading YOLOv8 model...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("✅ Model downloaded successfully.")


model = YOLO(MODEL_PATH)


denied_objects = ["cell phone", "laptop", "tv", "remote"]


OBJECT_DETECTION_TIME = 2.0  
detected_objects_timers = {}
alerted_objects = set()


cap = cv2.VideoCapture(0)
print("📷 Scanning for denied items... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detected_classes = [model.names[int(cls)] for cls in results.boxes.cls]

    for item in denied_objects:
        if item in detected_classes:
            if item not in detected_objects_timers:
                detected_objects_timers[item] = time.time()
                alerted_objects.discard(item)
            elif time.time() - detected_objects_timers[item] >= OBJECT_DETECTION_TIME:
                if item not in alerted_objects:
                    print(f"⚠️ Denied item detected: {item.upper()}")
                    alerted_objects.add(item)
        else:
            if item in detected_objects_timers:
                del detected_objects_timers[item]
                alerted_objects.discard(item)

    annotated_frame = results.plot()
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
