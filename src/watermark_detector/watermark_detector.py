import os
import tempfile
from PIL import Image
import cv2
from .predictor import WatermarksPredictor
from .utils import get_convnext_tiny_model

# Modelni yuklash
MODEL_PATH = "src/watermark_detector/models/convnext-tiny_watermarks_detector.pth"
MODEL, TRANSFORM = get_convnext_tiny_model(MODEL_PATH, fp16=False, device="cpu")
PREDICTOR = WatermarksPredictor(wm_model=MODEL, classifier_transforms=TRANSFORM, device="cpu")


def detect_watermark(image_path: str) -> str:
    """
    Rasmda watermark (suv belgisi) bor-yo‘qligini aniqlaydi.

    Args:
        image_path (str): Tekshiriladigan rasmning to‘liq yo‘li.

    Returns:
        str: `"✅ Watermark detected"` agar watermark aniqlansa,
             aks holda `"❌ No watermark"`.
    """
    img = Image.open(image_path).convert("RGB")
    result = PREDICTOR.predict_image(img)
    return "✅ Watermark detected" if result else "❌ No watermark"


def detect_watermark_in_video(video_path: str, frame_sample_rate: int = 10) -> dict:
    """
    Video ichidagi frame'lar orasidan har `frame_sample_rate`-chi frame'ni tekshiradi.
    Agar birorta frame'da watermark topilsa, `True` qaytaradi (early stop).

    Args:
        video_path (str): Tekshiriladigan video fayl yo‘li.
        frame_sample_rate (int, optional): Har necha frame'da bitta kadrni tekshirish. Default 10.

    Returns:
        dict: 
            {
                "watermark_detected": bool,  # Haqiqatan watermark topildimi?
                "message": str               # "✅ Watermark detected." yoki "❌ No watermark found in the video."
            }
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    detected = False
    with tempfile.TemporaryDirectory() as tmpdir:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_sample_rate == 0:
                frame_path = os.path.join(tmpdir, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, frame)

                image_pil = Image.open(frame_path).convert("RGB")
                result = PREDICTOR.predict_image(image_pil)

                if result:
                    detected = True
                    break  # early stop: topildi

            frame_idx += 1

    cap.release()
    return {
        "watermark_detected": detected,
        "message": (
            f"✅ Watermark detected."
            if detected else
            "❌ No watermark found in the video."
        )
    }
