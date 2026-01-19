from ultralytics import YOLO
import cv2
from datetime import datetime

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    exit()

animal_classes = ["dog", "cat", "cow", "horse", "sheep", "bird"]

print("📸 Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    total_objects = 0
    animal_count = 0

    with open("detection_report.txt", "a") as file:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                if confidence < 0.5:
                    continue

                total_objects += 1
                if class_name in animal_classes:
                    animal_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label = f"{class_name} {confidence*100:.1f}%"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                file.write(
                    f"[{timestamp}] Webcam | Object: {class_name} | Confidence: {confidence*100:.2f}%\n"
                )

    cv2.putText(frame, f"Objects: {total_objects}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Animals: {animal_count}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Real-Time Object & Animal Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("📸 Press 'q' to quit")


