from ultralytics import YOLO
import cv2
import time

model = YOLO('yolov8n.pt')  

denied_objects = ["cell phone", "laptop", "tv", "remote"]

# Time-based detection variables
OBJECT_DETECTION_TIME = 2.0  # 2 seconds
detected_objects_timers = {}  # Track each object's detection time
alerted_objects = set()  # Track which objects have been alerted

cap = cv2.VideoCapture(0)

print("Scanning for denied items... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detected_classes = [model.names[int(cls)] for cls in results.boxes.cls]

    # Check each denied object
    for item in denied_objects:
        if item in detected_classes:
            if item not in detected_objects_timers:
                detected_objects_timers[item] = time.time()  # Start timer
                alerted_objects.discard(item)  # Remove from alerted set
            elif time.time() - detected_objects_timers[item] >= OBJECT_DETECTION_TIME:
                if item not in alerted_objects:
                    print(f"⚠️ Denied item detected: {item.upper()}")
                    alerted_objects.add(item)  # Mark as alerted
        else:
            if item in detected_objects_timers:
                del detected_objects_timers[item]  # Remove timer when object disappears
                alerted_objects.discard(item)  # Remove from alerted set

    annotated_frame = results.plot()
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
