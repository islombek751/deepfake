from fastapi import FastAPI, UploadFile, File
from datetime import datetime
import os
import time
import psutil

from src.watermark_detector.watermark_detector import detect_watermark
from .fake_detector.detect import check_image_fake

app = FastAPI()

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/check-image/")
async def check_image(file: UploadFile = File(...)):
    process = psutil.Process()
    start_time = time.time()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    content = await file.read()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(content)

    ai_result = check_image_fake(content)
    watermark_result = detect_watermark(file_path)

    duration = time.time() - start_time
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    cpu_percent = process.cpu_percent(interval=0.1)

    return {
        "ai_generated_checker": ai_result,
        "watermark_result": watermark_result,
        "metrics": {
            "duration_seconds": round(duration, 3),
            "memory_used_MB": round(mem_after - mem_before, 2),
            "cpu_percent": cpu_percent,
        }
    }
