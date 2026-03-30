import torch
from ultralytics import YOLO
from torchvision import transforms
import cv2

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

size_convert = 640  # Kích thước chuẩn để đưa qua model
conf_thres = 0.4
iou_thres = 0.5

face_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),  # Input PIL => (3, 56, 56), /255.0
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

# Get the correct path to weights directory
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(current_dir, "..", "weights", "yolov8n-face.pt")
face_detector = YOLO(yolo_model_path)


@torch.no_grad()
def detect_face(image_face):
    # img = preprocess_image(image_face, size_convert)
    with torch.no_grad():
        result = face_detector.predict(image_face, conf=conf_thres, iou=iou_thres)
    return result
