"""
detect.py

This module provides a utility to check whether an image is AI-generated or real
using a pre-trained classification model.

Features:
- Automatically loads the latest model checkpoint from a directory.
- Applies consistent transforms to input images.
- Returns classification label and confidence score.
"""

import os
import glob
import torch
from PIL import Image
from io import BytesIO

from .model import get_model
from .custom_dataset import get_transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load transformation pipeline and label map
transform = get_transform()
label_map = {0: "real", 1: "fake"}


def load_latest_model(weights_folder: str = "src/fake_detector/models") -> torch.nn.Module:
    """
    Loads the latest model checkpoint from the given folder.

    Args:
        weights_folder (str): Directory where model .pth files are stored.

    Returns:
        torch.nn.Module: Loaded model in eval mode.

    Raises:
        FileNotFoundError: If no model files are found in the folder.
    """
    model_files = glob.glob(os.path.join(weights_folder, "model_epoch_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {weights_folder}.")

    latest = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest, map_location=device)

    model = get_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


# Load the model once globally
model = load_latest_model()


def check_image_fake(image_bytes: bytes) -> dict:
    """
    Checks if an image is real or AI-generated (fake).

    Args:
        image_bytes (bytes): The image file in bytes.

    Returns:
        dict: {
            "label": "real" or "fake",
            "confidence": float (percentage),
        }
        OR
        dict: {"error": str} on failure.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted = torch.argmax(probabilities).item()
            confidence = round(probabilities[predicted].item() * 100, 2)

        label_map = {0: "fake", 1: "real"}
        return {
            "label": label_map[predicted],
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}


    except Exception as e:
        return {
            "error": str(e)
        }
