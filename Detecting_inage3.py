from ultralytics import YOLO
import cv2
from datetime import datetime

model = YOLO("yolov8n.pt")

image = cv2.imread("images.jpeg")
if image is None:
    print("❌ Image not found!")
    exit()

results = model(image)

animal_classes = ["dog", "cat", "cow", "horse", "sheep", "bird"]

total_objects = 0
animal_count = 0

with open("detection_report.txt", "a") as file:
    file.write("\n--- New Detection Session ---\n")

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

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name} {confidence*100:.1f}%"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(
                f"[{timestamp}] Object: {class_name} | Confidence: {confidence*100:.2f}%\n"
            )

    file.write(f"Total Objects Detected: {total_objects}\n")
    file.write(f"Animals Detected: {animal_count}\n")

cv2.putText(image, f"Total Objects: {total_objects}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
cv2.putText(image, f"Animals: {animal_count}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.imshow("Day 3 - Detection with Report", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
