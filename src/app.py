import cv2
import numpy as np
import torch
import pickle
import sys
import os
import csv
from datetime import datetime
from flask import Flask, Response, request, jsonify
from PIL import Image, ImageDraw, ImageFont
import base64
import re
import warnings
import onnxruntime as ort
from flasgger import Swagger
import yaml


warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_models, detect_and_crop_face, extract_arcface_features
# from trainer import IncrementalMLPClassifier


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
swagger_path = os.path.join(BASE_DIR, "swagger.yml")

with open(swagger_path, "r", encoding="utf-8") as f:
    swagger_template = yaml.safe_load(f)

swagger = Swagger(app, template=swagger_template)
# ==============================
# Global variables for models
# ==============================
face_detector = None
arcface_model = None
onnx_session = None
label_encoder = None
device = None


def _resolve_model_paths():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(current_dir, "..", "models"),
        os.path.join(current_dir, "..", "model"),
    ]

    encoder_path = None
    onnx_model_path = None

    for base in candidates:
        e = os.path.join(base, "incremental_label_encoder.pkl")
        m = os.path.join(base, "incremental_mlp_model.onnx")
        if encoder_path is None and os.path.exists(e):
            encoder_path = e
        if onnx_model_path is None and os.path.exists(m):
            onnx_model_path = m

    return encoder_path, onnx_model_path


def load_recognition_models():

    global face_detector, arcface_model, onnx_session, label_encoder, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")

    print("[Init] Loading YOLO and ArcFace models...")

    face_detector, arcface_model = load_models()

    print("[Init] Loading incremental ONNX classifier + label encoder...")
    try:
        encoder_path, onnx_model_path = _resolve_model_paths()

        if not encoder_path:
            raise FileNotFoundError(
                "incremental_label_encoder.pkl not found in ../models or ../model"
            )
        if not onnx_model_path:
            raise FileNotFoundError(
                "incremental_mlp_model.onnx not found in ../models or ../model"
            )

        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)

        onnx_session = ort.InferenceSession(onnx_model_path)

        in_names = [i.name for i in onnx_session.get_inputs()]
        out_names = [o.name for o in onnx_session.get_outputs()]
        print(f"[Init] ONNX inputs: {in_names}")
        print(f"[Init] ONNX outputs: {out_names}")

        print("[Init] All models loaded successfully!")

    except Exception as e:
        print(f"[Init][Error] {e}")
        print("Please ensure models are exported to ONNX and encoder is saved.")
        return False

    return True


def reload_incremental_models():
    global onnx_session, label_encoder
    print("[Reload] Reloading incremental ONNX classifier + label encoder...")
    try:
        encoder_path, onnx_model_path = _resolve_model_paths()

        if not encoder_path or not os.path.exists(encoder_path):
            print("[Reload][Warning] Encoder not found.")
            return False
        if not onnx_model_path or not os.path.exists(onnx_model_path):
            print("[Reload][Warning] ONNX model not found.")
            return False

        with open(encoder_path, "rb") as f:
            new_label_encoder = pickle.load(f)

        new_onnx_session = ort.InferenceSession(onnx_model_path)

        # Atomically swap
        label_encoder = new_label_encoder
        onnx_session = new_onnx_session
        print("[Reload] Models reloaded successfully!")
        return True
    except Exception as e:
        print(f"[Reload][Error] {e}")
        return False


def recognize_face(face_features: np.ndarray):

    global onnx_session, label_encoder

    if onnx_session is None or label_encoder is None:
        raise RuntimeError("Models not loaded. Call load_recognition_models() first.")

    feats = face_features.reshape(1, -1).astype(np.float32)

    input_name = onnx_session.get_inputs()[0].name

    outputs = onnx_session.run(None, {input_name: feats})
    logits = outputs[0]  # shape (1, num_classes)

    # Softmax thủ công để ra xác suất
    # trừ max để ổn định số học
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    pred_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    predicted_label = label_encoder.inverse_transform([pred_idx])[0]
    return predicted_label, confidence


def process_image_for_recognition(image_array: np.ndarray):

    global face_detector, arcface_model

    results = face_detector.predict(image_array, conf=0.4, iou=0.5)

    faces_found = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # Crop face
        face_crop = image_array[y1:y2, x1:x2]

        # Extract ArcFace features
        try:
            face_features = extract_arcface_features(face_crop, arcface_model)

            predicted_label, confidence = recognize_face(face_features)

            faces_found.append(
                {
                    "label": predicted_label,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2],
                }
            )

        except Exception as e:
            print(f"[Recognition][Warn] Error processing a face: {e}")
            continue

    return faces_found


@app.route("/verify-employee", methods=["POST"])
def verify_employee():

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        imgs = data.get("imgs", [])
        seconds_time = data.get("secondsTime")
        time_verify = data.get("timeVerify")

        if not imgs:
            return jsonify({"error": "No images provided"}), 400

        # Giải mã ảnh đầu tiên (dataURL hoặc thuần base64)
        img_b64 = re.sub(r"^data:image\/[a-zA-Z]+;base64,", "", imgs[0])
        img_data = base64.b64decode(img_b64)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        faces_found = process_image_for_recognition(image)
        if not faces_found:
            return jsonify({"partId": None, "probability": "0", "emotion": "Neutral"})

        # Lấy mặt có confidence cao nhất
        best_face = max(faces_found, key=lambda x: x["confidence"])

        part_id = best_face["label"]
        probability = f"{best_face['confidence']:.4f}"
        emotion = "Happy" if best_face["confidence"] > 0.8 else "Neutral"

        return jsonify(
            {"partId": part_id, "probability": probability, "emotion": emotion}
        )

    except Exception as e:
        print(f"[API][Error] /verify-employee: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/train", methods=["POST"])
def train_model():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        employee_id = data.get("employeeId", "").strip()
        employee_name = data.get("employeeName", "").strip()
        seconds_time = data.get("secondsTime")
        imgs = data.get("imgs", [])

        if not employee_id or not imgs:
            return jsonify({"error": "employeeId and imgs are required"}), 400

        # Bỏ @ncc.asia nếu có
        employee_id = employee_id.replace("@ncc.asia", "")

        # Tạo folder dataset/<employeeId>
        dataset_dir = os.path.join(BASE_DIR, "dataset", employee_id)
        os.makedirs(dataset_dir, exist_ok=True)

        # Giải mã và lưu ảnh
        for idx, img_b64 in enumerate(imgs):
            try:
                img_clean = re.sub(r"^data:image\/[a-zA-Z]+;base64,", "", img_b64)
                img_data = base64.b64decode(img_clean)
                img_array = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[Warn] Image {idx} decode failed")
                    continue

                # Lưu ảnh với timestamp để tránh trùng tên
                filename = (
                    f"{employee_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                )
                cv2.imwrite(os.path.join(dataset_dir, filename), img)

            except Exception as e:
                print(f"[Warn] Error saving image {idx}: {e}")
                continue

        print(f"[API] Saved {len(imgs)} images for {employee_id}")

        # Gọi hai phần train cũ
        from extractor import main as extract_main
        from trainer import main as train_main

        extract_main()
        print("[API] Feature extraction completed. Starting model training...")

        train_main()
        print("[API] Model training completed successfully.")

        # Reload models so the new classes take effect
        reload_incremental_models()

        return jsonify({"message": "Training finished"}), 200

    except Exception as e:
        print(f"[API][Error] /train: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/train_remove_employee", methods=["POST"])
def train_remove_employee():
    from extractor import IncrementalFeatureExtractor

    try:
        data = request.get_json()
        employee_ids = data.get("employeeIds", [])

        if not employee_ids:
            return jsonify({"error": "employeeIds is required"}), 400

        # Khởi tạo extractor và load dữ liệu cũ
        extractor = IncrementalFeatureExtractor()

        # Gọi hàm xóa
        extractor.remove_employees(employee_ids)

        # Re-train model to apply deletion
        from trainer import main as train_main

        print("[API] Running train_main to update model after removal...")
        train_main()

        # Reload updated model
        reload_incremental_models()

        return jsonify({"status": "success", "removed": employee_ids}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Backend AI Service is starting...")

    if not load_recognition_models():
        print("Failed to load recognition models.")
        sys.exit(1)

    host = os.getenv("APP_HOST", "0.0.0.0")
    app.run(host=host, port=5000, threaded=True)
