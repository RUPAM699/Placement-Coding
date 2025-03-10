import cv2
from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break 
    frame = resize(frame, height=500)
    rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects_np = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects_np, probs=None, overlapThresh=0.65)

    for (xa, ya, xb, yb) in pick:
        cv2.rectangle(frame, (xa, ya), (xb, yb), (0, 255, 0), 2)

    cv2.imshow('People Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()