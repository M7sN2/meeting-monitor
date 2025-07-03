from ultralytics import YOLO
import cv2


model = YOLO('yolov8n.pt')  


denied_objects = ["cell phone", "laptop", "tv", "remote"]


cap = cv2.VideoCapture(0)

print("Scanning for denied items... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)[0]

    
    detected_classes = [model.names[int(cls)] for cls in results.boxes.cls]

    
    for item in denied_objects:
        if item in detected_classes:
            print(f"⚠️ Denied item detected: {item.upper()}")

    
    annotated_frame = results.plot()
    cv2.imshow("Detection", annotated_frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
