import cv2
import time
from ultralytics import YOLO
from facenet_feature import extract_feature_vector

# Load YOLO model
model = YOLO("models/l_version_1_300.pt")

classNames = ["fake", "real"]
confidence_threshold = 0.9

cap = cv2.VideoCapture(0)

real_counter = 0
captured = False

while True:

    success, img = cap.read()
    if not success:
        break

    results = model(img)

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            label = classNames[cls]

            # -------- Stability Logic --------
            if label == "real" and conf > confidence_threshold:
                real_counter += 1
            else:
                real_counter = 0

            # -------- Capture Condition --------
            if real_counter >= 5 and not captured:

                print("✅ REAL detected!")

                face = img[y1:y2, x1:x2]
                image_path = "captured_face.jpg"
                cv2.imwrite(image_path, face)

                print("📸 Image Captured!")

                embedding = extract_feature_vector(image_path)

                print("🔢 Feature Vector Extracted!")
                print("Length:", len(embedding))
                print("First 5 values:", embedding[:5])

                captured = True

                # Break inner loop
                break

            # -------- Draw Bounding Box --------
            color = (0, 255, 0) if label == "real" else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cv2.putText(img,
                        f"{label.upper()} {int(conf * 100)}%",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        if captured:
            break

    cv2.imshow("Liveness + Feature Extraction", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if captured:
        break

cap.release()
cv2.destroyAllWindows()