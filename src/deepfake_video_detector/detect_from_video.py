import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
import numpy as np
from .models import model_selection
from .transform import xception_default_data_transforms


_face_detector = dlib.get_frontal_face_detector()
_model = None

def load_model_once(model_path, use_cuda=True):
    global _model
    if _model is None:
        model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
        model.load_state_dict(torch.load(model_path, map_location="cuda" if use_cuda else "cpu"))
        if use_cuda:
            model = model.cuda()
        model.eval()
        _model = model
    return _model

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize and size_bb < minsize:
        size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image)).unsqueeze(0)
    return preprocessed_image.cuda() if cuda else preprocessed_image


def predict_with_model(image, model, cuda=True):
    input_tensor = preprocess_image(image, cuda)
    with torch.no_grad():
        output = model(input_tensor)
        output = nn.Softmax(dim=1)(output)
        prediction = torch.argmax(output, dim=1).item()
        confidence = output[0][1].item()  # fake ehtimoli
    return prediction, confidence


def analyze_video(video_path: str, model_path="src/deepfake_video_detector/models/ffpp_c23.pth", use_cuda=True) -> str:
    model = load_model_once(model_path, use_cuda=use_cuda)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    predictions = []

    while cap.isOpened() and frame_count < 50:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _face_detector(gray, 1)
        if not faces:
            continue
        face = faces[0]
        x, y, size = get_boundingbox(face, frame.shape[1], frame.shape[0])
        cropped_face = frame[y:y+size, x:x+size]
        pred, conf = predict_with_model(cropped_face, model, cuda=use_cuda)
        predictions.append(conf)

    cap.release()

    if not predictions:
        return "unknown"

    avg_score = np.mean(predictions)
    return "fake" if avg_score > 0.5 else "real"
