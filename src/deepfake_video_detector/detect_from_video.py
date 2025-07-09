import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
import numpy as np
from .models import model_selection
from .transform import xception_default_data_transforms

# Global dlib yuz aniqlovchi
_face_detector = dlib.get_frontal_face_detector()
_model = None

def load_model_once(model_path):
    """
    Load the deepfake detection model once and cache it globally.

    Parameters:
        model_path (str): Path to the model .pth file.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    global _model
    if _model is None:
        model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)

        # GPU mavjud bo‘lsa — 'cuda', bo‘lmasa — 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        _model = model
    return _model


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Compute a square bounding box around the detected face with optional scaling.

    Parameters:
        face (dlib.rectangle): Face bounding box from dlib detector.
        width (int): Width of the original frame.
        height (int): Height of the original frame.
        scale (float): Scaling factor for the bounding box size.
        minsize (int or None): Minimum bounding box size.

    Returns:
        tuple: (x1, y1, size) — top-left coordinates and box size.
    """
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


def preprocess_image(image, cuda=None):
    """
    Preprocess an image for model inference.

    Converts image to RGB, applies Xception transforms, and adds batch dimension.

    Parameters:
        image (np.ndarray): BGR image (OpenCV format).
        cuda (bool): Whether to move tensor to GPU.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    if cuda is None:
        cuda = torch.cuda.is_available()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image)).unsqueeze(0)
    return preprocessed_image.cuda() if cuda else preprocessed_image


def predict_with_model(image, model):
    """
    Predict whether an image is real or fake using the given model.

    Parameters:
        image (np.ndarray): Cropped face image.
        model (torch.nn.Module): Loaded deepfake detection model.

    Returns:
        tuple:
            - prediction (int): 0 for real, 1 for fake.
            - confidence (float): Probability of being fake (0 to 1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = preprocess_image(image).to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        output = torch.nn.functional.softmax(output, dim=1)
        prediction = torch.argmax(output, dim=1).item()
        confidence = output[0][1].item()  # Probability that it's fake
    return prediction, confidence


def analyze_video(video_path: str, model_path="src/deepfake_video_detector/models/ffpp_c23.pth") -> str:
    """
    Analyze a video to detect if it's real or deepfake.

    This function reads up to 50 frames from the video, detects a face in each frame,
    crops and preprocesses the face, then uses a neural network to predict fake/real.
    It averages the predictions to determine the final result.

    Parameters:
        video_path (str): Path to the input video file.
        model_path (str): Path to the model weights (.pth file).
        use_cuda (bool): Whether to use GPU for inference.

    Returns:
        str: "real", "fake", or "unknown" (if no faces found or no frames processed).
    """
    model = load_model_once(model_path)
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
        for face in faces:
            x, y, size = get_boundingbox(face, frame.shape[1], frame.shape[0])
            cropped_face = frame[y:y+size, x:x+size]

            pred, conf = predict_with_model(cropped_face, model)
            predictions.append(conf)

    cap.release()

    if not predictions:
        return "unknown"
    avg_score = np.mean(predictions)
    return "fake" if avg_score > 0.5 else "real"
