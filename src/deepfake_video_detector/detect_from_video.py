import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
import numpy as np
import mediapipe as mp

from .models import model_selection
from .transform import xception_default_data_transforms

# Global mediapipe yuz aniqlovchi
mp_face_detection = mp.solutions.face_detection
_face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
_model = None


def load_model_once(model_path):
    global _model
    if _model is None:
        model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _model = model
    return _model


def preprocess_image(image, cuda=None):
    if cuda is None:
        cuda = torch.cuda.is_available()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image)).unsqueeze(0)
    return preprocessed_image.cuda() if cuda else preprocessed_image


def predict_with_model(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess_image(image).to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = output[0][1].item()
    return prediction, confidence


def get_boundingbox_mediapipe(frame):
    results = _face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        ih, iw, _ = frame.shape
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            x = max(int(bbox.xmin * iw), 0)
            y = max(int(bbox.ymin * ih), 0)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            faces.append((x, y, w, h))
    return faces


def analyze_video(video_path: str, model_path="src/deepfake_video_detector/models/ffpp_c23.pth") -> str:
    model = load_model_once(model_path)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while cap.isOpened() and frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        faces = get_boundingbox_mediapipe(frame)
        for x, y, w, h in faces:
            cropped_face = frame[y:y+h, x:x+w]
            if cropped_face.size == 0:
                continue

            pred, conf = predict_with_model(cropped_face, model)
            predictions.append(conf)

    cap.release()

    if not predictions:
        return "unknown"
    avg_score = np.mean(predictions)
    return "fake" if avg_score > 0.5 else "real"
