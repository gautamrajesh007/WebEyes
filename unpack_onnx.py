import cv2
import time
import numpy as np

model: cv2.dnn.Net = cv2.dnn.readNetFromONNX("./Models/Onnx_models/yolov8n.onnx")
names = "person".split(";")

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()

    height, width, _ = frame.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = frame
    scale = length / 640

    # First run to 'warm-up' the model
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    model.forward()

    # Second run
    t1 = time.monotonic()
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()
    print("dT:", time.monotonic() - t1)

    # Show results
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []
    output = outputs[0]
    for i in range(rows):
        classes_scores = output[i][4:]
        minScore, maxScore, minClassLoc, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25 and names[maxClassIndex] == "person":
            box = [output[i][0] - 0.5 * output[i][2], output[i][1] - 0.5 * output[i][3],
                   output[i][2], output[i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    for index in result_boxes:
        box = boxes[index]
        box_out = [round(box[0]*scale), round(box[1]*scale),
                   round((box[0] + box[2])*scale), round((box[1] + box[3])*scale)]
        print("Rect:", names[class_ids[index]], scores[index], box_out)

    cv2.imshow("Webcam", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
