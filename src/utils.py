import os
import cv2
import numpy as np
import torch
import pickle
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Import from detect and embedding_face modules
from detect import detect_face, face_detector
from embedding_face import get_feature, ort_session as arcface_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_models():
    """Load YOLOv8 and ArcFace models"""
    # Models are already loaded globally in detect.py and embedding_face.py
    return face_detector, arcface_model


def detect_and_crop_face(image, face_detector):
    """Detect and crop face from image"""
    results = detect_face(image)

    if len(results[0].boxes.xyxy) == 0:
        return None

    # Get the first detected face
    x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[0])
    face_crop = image[y1:y2, x1:x2]

    return face_crop


def extract_arcface_features(face_image, arcface_model):
    """Extract ArcFace features from face image"""
    return get_feature(face_image).flatten()


def print_class_distribution(labels):
    """Print class distribution"""
    unique_labels, counts = np.unique(labels, return_counts=True)

    print("\nClass Distribution:")
    print("-" * 50)
    for label, count in zip(unique_labels, counts):
        print(f"{label}: {count} images")
    print("-" * 50)
    print(f"Total classes: {len(unique_labels)}")
    print(f"Total images: {len(labels)}")

    return unique_labels, counts
