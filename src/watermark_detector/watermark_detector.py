from PIL import Image
from .predictor import WatermarksPredictor
from .utils import get_convnext_tiny_model

MODEL_PATH = "src/watermark_detector/models/convnext-tiny_watermarks_detector.pth"
MODEL, TRANSFORM = get_convnext_tiny_model(MODEL_PATH, fp16=False, device="cpu")
PREDICTOR = WatermarksPredictor(wm_model=MODEL, classifier_transforms=TRANSFORM, device="cpu")


def detect_watermark(image_path: str) -> str:
    """
    Bitta rasmda watermark bor-yo‘qligini aniqlaydi.
    Model va transform oldindan yuklangan.
    """
    img = Image.open(image_path).convert("RGB")
    result = PREDICTOR.predict_image(img)
    return "✅ Watermark detected" if result else "❌ No watermark"
