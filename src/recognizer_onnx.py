import cv2
import numpy as np
import sys
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import warnings
import pickle
import time
import threading
import queue    
from collections import deque 
import onnxruntime as ort
warnings.filterwarnings('ignore')


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_models, detect_and_crop_face, extract_arcface_features
# from incremental_model_training import IncrementalMLPClassifier

class IncrementalFaceRecognition:
    def __init__(self, confidence_threshold=0.7, ai_input_size=(320, 240),
                 thread_queue_size=5, camera_source=0):
        """Initialize incremental face recognition system"""
        self.confidence_threshold = confidence_threshold
        self.ai_input_size = ai_input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print("Loading models...")
        self.face_detector, self.arcface_model = load_models()
        
        print("Loading incremental MLP classifier...")
        try:
            self.ort_session, self.label_encoder = self.load_incremental_onnx_model()
            print("Models loaded successfully!")
        except FileNotFoundError:
            print("Incremental model not found. Please run incremental_model_training.py first.")
            sys.exit(1)
        
     
        self.cap = cv2.VideoCapture(camera_source)
        
        if not self.cap.isOpened():
            print("Không thể mở camera")
            sys.exit(1)

        
        self.frame_queue = queue.Queue(maxsize=thread_queue_size)

        self.last_results = deque(maxlen=1)
        self.last_results.append([]) 

        print(f"Recognition threshold: {confidence_threshold}")
        print(f"AI input size: {self.ai_input_size[0]}x{self.ai_input_size[1]}")
        print("Press 'q' to quit")
    
    # def load_incremental_model(self, model_path="../models/incremental_mlp_model.pth", 
    #                           encoder_path="../models/incremental_label_encoder.pkl"):
    #     """Load incremental model and encoder"""
    #     current_dir = os.path.dirname(os.path.abspath(__file__))
    #     model_path = os.path.join(current_dir, "..", "models", "incremental_mlp_model.pth")
    #     encoder_path = os.path.join(current_dir, "..", "models", "incremental_label_encoder.pkl")
        
    #     # Load encoder
    #     with open(encoder_path, 'rb') as f:
    #         label_encoder = pickle.load(f)
        
    #     # Create model with same architecture
    #     input_dim = 512  # ArcFace feature dimension
    #     num_classes = len(label_encoder.classes_)
    #     model = IncrementalMLPClassifier(input_dim, num_classes)
        
    #     # Load state dict
    #     model.load_state_dict(torch.load(model_path, map_location=self.device))
    #     model.to(self.device)
    #     model.eval()
        
    #     return model, label_encoder
    
    def load_incremental_onnx_model(self):
        """Load ONNX model and label encoder"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        

        encoder_path = os.path.join(current_dir, "..", "models", "incremental_label_encoder.pkl")
        onnx_model_path = os.path.join(current_dir, "..", "models", "incremental_mlp_model.onnx")

     
        if not os.path.exists(encoder_path):
            encoder_path = os.path.join(current_dir, "..", "model", "incremental_label_encoder.pkl")
        if not os.path.exists(onnx_model_path):
            onnx_model_path = os.path.join(current_dir, "..", "model", "incremental_mlp_model.onnx")

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Not found encoder file at {encoder_path}")
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Not found ONNX model file at {onnx_model_path}")
        
     
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
     
        ort_session = ort.InferenceSession(onnx_model_path)
        
        return ort_session, label_encoder
    
    def put_vietnamese_text(self, img, text, position, font_size=20, color=(0, 255, 0)):
        """Draw Vietnamese text on image"""
       
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
      
        try:
            font_paths = [
                "arial.ttf",
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/tahoma.ttf",
                "C:/Windows/Fonts/calibri.ttf"
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
    
    # def recognize_face(self, face_features):

    #     """Recognize face using incremental MLP ONNX model"""

    #     input_data = face_features.reshape(1, -1).astype(np.float32)

     
    #     input_name = self.ort_session.get_inputs()[0].name
        
     
    #     outputs = self.ort_session.run(None, {input_name: input_data})
    #     predictions = outputs[0]

      
    #     exp_predictions = np.exp(predictions - np.max(predictions))
    #     probabilities = exp_predictions / np.sum(exp_predictions, axis=1, keepdims=True)
        
      
    #     predicted_class = np.argmax(probabilities, axis=1)[0]
    #     confidence = np.max(probabilities)
        

    #     predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
    #     return predicted_label, confidence
    
    def recognize_face(self, face_features):
        """Recognize face using incremental MLP ONNX model"""
        # Chuẩn hóa input cho ONNX
        input_data = face_features.reshape(1, -1).astype(np.float32)

        input_name = self.ort_session.get_inputs()[0].name

        # Chạy model -> logits
        outputs = self.ort_session.run(None, {input_name: input_data})
        logits = outputs[0]  # shape: (batch_size, num_classes)

        # Softmax ổn định số học, hỗ trợ batch > 1
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Lấy class và confidence cho sample đầu tiên (batch=1)
        predicted_class = int(np.argmax(probabilities, axis=1)[0])
        confidence = float(np.max(probabilities, axis=1)[0])

        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]

        return predicted_label, confidence

    def process_frame(self, frame):
        """Process a single frame"""

        # Resize input frame for AI model
        original_h, original_w, _ = frame.shape
        ai_w, ai_h = self.ai_input_size

        # Resizing the frame for faster AI processing
        resized_frame = cv2.resize(frame, self.ai_input_size)

        # Detect faces
        results = self.face_detector.predict(resized_frame, conf=0.4, iou=0.5)

        processed_results = []
        for box in results[0].boxes.xyxy:
            x1_resized, y1_resized, x2_resized, y2_resized = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            
            
            face_crop = resized_frame[y1_resized:y2_resized, x1_resized:x2_resized]
            
           
            try:
                face_features = extract_arcface_features(face_crop, self.arcface_model)
                
              
                predicted_label, confidence = self.recognize_face(face_features)

               
                scale_x = original_w / ai_w
                scale_y = original_h / ai_h
                x1, y1 = int(x1_resized * scale_x), int(y1_resized * scale_y)
                x2, y2 = int(x2_resized * scale_x), int(y2_resized * scale_y)
                
                processed_results.append({
                    "box": (x1, y1, x2, y2),
                    "label": predicted_label,
                    "confidence": confidence
                })
                    
            except Exception as e:
                print(f"Error in processing: {e}")
        
        self.last_results.append(processed_results)
        return frame
    
    def worker(self):
        while True:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                
                while not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                
                self.process_frame(frame)
                self.frame_queue.task_done()
            except queue.Empty:
                continue
    
    def run(self):
        """Run real-time recognition"""
        print("Starting incremental face recognition...")
        
        worker_thread = threading.Thread(target=self.worker, daemon=True)
        worker_thread.start()
        
        fps_start_time = time.time()
        frame_counter = 0
        fps = 0.0
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Can not read frames from camera")
                break
            # Push frame in queue for AI to process
            if self.frame_queue.empty():
                try:
                    self.frame_queue.put(frame.copy(), block=False) # Use .copy() to prevent conflict data
                except queue.Full:
                    pass # Pass if queue is full

            # Get latest detection result
            results = self.last_results[-1]
            

            for result in results:
                x1, y1, x2, y2 = result["box"]
                confidence = result["confidence"]
                
                if confidence > self.confidence_threshold:
                    label = f"{result['label']} ({confidence:.2f})"
                    color = (0, 255, 0)
                else:
                    label = f"Unknown ({confidence:.2f})"
                    color = (0, 0, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                frame = self.put_vietnamese_text(frame, label, (x1, y1 - 30), 20, color)
            

            frame_counter += 1
            if frame_counter % 30 == 0:
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                print(f"Display FPS: {fps:.2f}")
                fps_start_time = time.time()

            cv2.putText(frame, f"Display FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize to show 
            processed_frame = cv2.resize(frame, (960, 720))
            cv2.imshow("Incremental Face Recognition", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        

        self.frame_queue.put(None)
        worker_thread.join()
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("Recognition stopped.")

def main():
    """Main function"""
    print("=" * 60)
    print("INCREMENTAL FACE RECOGNITION WITH MLP")
    print("=" * 60)
    
    # Create recognition system
    recognizer = IncrementalFaceRecognition(confidence_threshold=0.7, ai_input_size=(320, 240))
    
    # Run recognition
    recognizer.run()

if __name__ == "__main__":
    main() 