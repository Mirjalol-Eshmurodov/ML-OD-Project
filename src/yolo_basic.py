# Yolo for image
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
results = model("/Users/eshmurodov.m/Desktop/ZAKO/Projects/ML-OD-Project/data/image/bikes_in_park.jpg", show =True)
img = results[0].plot()
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

