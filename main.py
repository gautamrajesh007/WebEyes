import cv2
import numpy as np

cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (640, 480), fx=0, fy=0, interpolation=cv2.INTER_AREA)

    # cv2.imshow('Frame', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow('Thresh', thresh)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cap.destroyAllWindows()
