from ultralytics import YOLO
import cv2
import time

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize variables for FPS calculation
prev_frame_time = 0
fps_counter = 0

# model
model = YOLO("Models/Pt_models/yolov8n.pt")

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Calculate FPS
    current_time = time.time()
    elapsed_time = current_time - prev_frame_time
    prev_frame_time = current_time

    fps = 1 / elapsed_time

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])

            # Only process "person" class
            if cls == 0:  # 0 corresponds to the "person" class
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw rectangle with FPS
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                fps_text = f"FPS: {fps:.2f}"
                cv2.putText(img, fps_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imshow('Webcam (People Detection)', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
