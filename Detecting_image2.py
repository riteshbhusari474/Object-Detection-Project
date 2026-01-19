from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

image = cv2.imread("images.jpeg")
if image is None:
    print("❌ Image not found!")
    exit()

results = model(image)

total_objects = 0
animal_count = 0

animal_classes = ["dog", "cat", "cow", "horse", "sheep", "bird"]

for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if confidence >= 0.5:
            continue

        total_objects += 1

        if class_name in animal_classes:
            animal_count += 1

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence*100:.1f}%"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.putText(image, f"Total Objects: {total_objects}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.putText(image, f"Animals Detected: {animal_count}", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

cv2.imshow("Day 2 - Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
