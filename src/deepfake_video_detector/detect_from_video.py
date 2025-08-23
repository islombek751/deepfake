import cv2
import numpy as np
from pathlib import Path

from src.fake_detector.detect import check_image_fake

def analyze_video(video_path: str, max_frames: int = 50) -> str:
    """
    Videoni frame bo'yicha tekshiradi va deepfake yoki real ekanligini aniqlaydi.
    
    Args:
        video_path (str): video fayl manzili
        max_frames (int): tekshiriladigan maksimal frame soni
    
    Returns:
        str: "real", "fake" yoki "unknown"
    """
    video_path = Path(video_path)
    if not video_path.exists():
        return "unknown"
    
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    predictions = []

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # OpenCV frame -> bytes
        _, buffer = cv2.imencode(".jpg", rgb_frame)
        frame_bytes = buffer.tobytes()

        # check_image_fake ni chaqirish
        result = check_image_fake(frame_bytes)

        # Real va fake ehtimollarini saqlash
        predictions.append({"real": result["real"], "fake": result["fake"]})

    cap.release()

    if not predictions:
        return "unknown"
    
    # O'rtacha ehtimollar
    avg_real = np.mean([p["real"] for p in predictions])
    avg_fake = np.mean([p["fake"] for p in predictions])

    return "fake" if avg_fake > avg_real else "real"
