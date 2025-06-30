import os
import glob
import torch
from PIL import Image
from io import BytesIO

from .model import get_model
from .custom_dataset import get_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transform()
label_map = {0: "real", 1: "fake"}

def load_latest_model(weights_folder="src/fake_detector/models"):
    model_files = glob.glob(os.path.join(weights_folder, "model_epoch_*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {weights_folder}.")
    
    latest = max(model_files, key=os.path.getctime)
    checkpoint = torch.load(latest, map_location=device)

    model = get_model(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

model = load_latest_model()

def check_image_fake(image_bytes: bytes) -> dict:
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted = torch.argmax(probabilities).item()

        return {
            "label": label_map[predicted],
            "confidence": round(probabilities[predicted].item() * 100, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }
