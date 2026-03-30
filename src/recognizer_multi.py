import cv2
import numpy as np
import sys
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import warnings
import pickle

warnings.filterwarnings("ignore")
from multiprocessing import Process, Queue
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_models, detect_and_crop_face, extract_arcface_features
from incremental_model_training import IncrementalMLPClassifier


def put_vietnamese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """Draw Vietnamese text on image"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font_paths = [
            "arial.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/tahoma.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Font error: {e}")
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


def camera_reader(q, camera_urls, frame_interval=5):
    """Camera reader process for multiprocessing"""
    cap = None
    for i, url in enumerate(camera_urls):
        print(f"Đang thử kết nối camera {i + 1}: {url}")
        if url == "0":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print(f" Kết nối thành công với camera: {url}")
            break
        else:
            print(f" Không thể kết nối với camera: {url}")
            if cap:
                cap.release()
    if not cap or not cap.isOpened():
        print(" Không thể kết nối với bất kỳ camera nào!")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame, thử kết nối lại...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(camera_urls[0], cv2.CAP_FFMPEG)
            continue
        if (frame_count % frame_interval) == 0:
            if not q.full():
                q.put(frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


def load_incremental_model(
    model_path="../models/incremental_mlp_model.pth",
    encoder_path="../models/incremental_label_encoder.pkl",
):
    """Load incremental model and encoder"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "models", "incremental_mlp_model.pth")
    encoder_path = os.path.join(
        current_dir, "..", "models", "incremental_label_encoder.pkl"
    )

    # Load encoder
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Create model with same architecture
    input_dim = 512  # ArcFace feature dimension
    num_classes = len(label_encoder.classes_)
    model = IncrementalMLPClassifier(input_dim, num_classes)

    # Load state dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, label_encoder


def ai_worker(q, confidence_threshold=0.7):
    """AI worker process for multiprocessing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading models...")
    face_detector, arcface_model = load_models()
    print("Loading incremental MLP classifier...")
    try:
        mlp_model, label_encoder = load_incremental_model()
        print("Models loaded successfully!")
    except FileNotFoundError:
        print(
            "Incremental model not found. Please run incremental_model_training.py first."
        )
        return

    mlp_model.eval()
    print("AI worker ready.")

    while True:
        frame = None
        # Lấy frame mới nhất trong queue (bỏ qua frame cũ nếu queue có nhiều frame)
        while not q.empty():
            frame = q.get()
        if frame is None:
            time.sleep(0.01)
            continue

        results = face_detector.predict(frame, conf=0.4, iou=0.5)
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face_crop = frame[y1:y2, x1:x2]
            try:
                face_features = extract_arcface_features(face_crop, arcface_model)
                features_tensor = torch.FloatTensor(face_features.reshape(1, -1)).to(
                    device
                )
                with torch.no_grad():
                    outputs = mlp_model(features_tensor)
                    predictions = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(predictions, dim=1).item()
                confidence = torch.max(predictions).item()
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                if confidence > confidence_threshold:
                    label = f"{predicted_label} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({confidence:.2f})"
                    color = (0, 0, 255)
            except Exception as e:
                label = "Error"
                color = (0, 0, 255)

            # Draw text and rectangle
            frame = put_vietnamese_text(frame, label, (x1, y1 - 30), 18, color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame = cv2.resize(frame, (960, 720))
        cv2.imshow("Incremental Face Recognition - Multiprocess", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_urls = [
        # "rtsp://admin:password@[IP_ADDRESS]:554/Streamming/channels/101",
        "0"
    ]
    q = Queue(maxsize=10)
    p1 = Process(target=camera_reader, args=(q, camera_urls))
    p2 = Process(target=ai_worker, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
