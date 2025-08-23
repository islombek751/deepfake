#Manba: https://www.kaggle.com/refs/hf-model/prithivMLmods/Deep-Fake-Detector-v2-Model

from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import io

# Model va processor ni yuklash
model_name = "prithivMLmods/deepfake-detector-model-v1"  
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Label mapping
id2label = {
    "0": "real",
    "1": "fake"
}

def check_image_fake(image_bytes: bytes) -> dict:
    """
    Rasmni bytes formatida qabul qiladi va uni deepfake yoki real ekanligini aniqlaydi.
    
    Args:
        image_bytes (bytes): Rasmning bytes ko'rinishi
    
    Returns:
        dict: {
            "real": 0.123,
            "fake": 0.877,
            "prediction": "fake"  # eng yuqori ehtimollikdagi label
        }
    """
    # Bytes dan PIL Image yaratish
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Modelga mos input yaratish
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Probability dict
    prediction_probs = {id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    # Eng yuqori ehtimollikdagi label
    predicted_label = max(prediction_probs, key=prediction_probs.get)
    prediction_probs["prediction"] = predicted_label
    
    return prediction_probs
