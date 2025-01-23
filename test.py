import os
from ultralytics import YOLO


model = YOLO("yolov8s.pt")
model.check_data("D:/Year4/4907/Training/C2A_Dataset/data.yaml")
