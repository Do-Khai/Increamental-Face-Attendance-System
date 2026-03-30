# pytorch
import torch
from PIL import Image
from torchvision import transforms
import torchvision

# other lib
import sys
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import onnxruntime as ort
from detect import detect_face
# model_embedded_face (insightface)
# from iresnet import iresnet100

# Get the correct path to weights directory
current_dir = os.path.dirname(os.path.abspath(__file__))
arcface_model_path = os.path.join(current_dir, "..", "weights", "arcface_r100.onnx")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = iresnet100(pretrained=False)
# model.load_state_dict(torch.load(arcface_model_path, map_location=device))
# model.eval()
# model.to(device)
ort_session = ort.InferenceSession(arcface_model_path)

face_preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@torch.no_grad()
def get_feature(face_image):
    """
    Extract features from a face image.

    Args:
        face_image: The input face image.

    Returns:
        numpy.ndarray: The extracted features.
    """
    # face_preprocess = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Resize((112, 112)),
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ]
    # )
    # Convert to RGB
    # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # # Preprocess image (BGR)
    # face_image = face_preprocess(face_image).unsqueeze(0).to(device)
    # with torch.no_grad():
    #     # Inference to get feature
    #     emb_img_face = model(face_image).cpu().numpy()

    # # Convert to array
    # images_emb = emb_img_face / np.linalg.norm(emb_img_face)

    # return images_emb
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Preprocess image and convert to a numpy array for ONNX
    face_tensor = face_preprocess(face_image).unsqueeze(0)
    face_numpy = face_tensor.cpu().numpy().astype(np.float32)

    # Get the name of the input tensor from the ONNX session
    input_name = ort_session.get_inputs()[0].name

    # Run inference using the ONNX session
    onnx_output = ort_session.run(None, {input_name: face_numpy})[0]

    # Convert to array and normalize
    images_emb = onnx_output / np.linalg.norm(onnx_output)

    return images_emb
