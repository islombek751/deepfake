import shutil
import uuid
import pynvml
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import time
import psutil
from src.deepfake_video_detector.detect_from_video import analyze_video

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


def get_gpu_info():
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_name": pynvml.nvmlDeviceGetName(handle).decode(),
            "gpu_memory_used_mb": round(memory.used / 1024 / 1024, 2),
            "gpu_memory_total_mb": round(memory.total / 1024 / 1024, 2),
            "gpu_utilization_percent": utilization.gpu
        }
    except Exception as e:
        return {"gpu": "not available", "error": str(e)}


@app.post("/check-video/")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    file_path = f"uploaded_videos/{file_id}_{file.filename}"
    os.makedirs("uploaded_videos", exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process = psutil.Process(os.getpid())
    start_time = time.time()
    cpu_start = process.cpu_times()
    mem_start = process.memory_info().rss

    gpu_before = get_gpu_info()

    ai_generated_check = analyze_video(video_path=file_path, use_cuda=True)

    gpu_after = get_gpu_info()

    elapsed_time = time.time() - start_time
    cpu_end = process.cpu_times()
    mem_end = process.memory_info().rss

    result = {
        "ai_generated_checker": ai_generated_check,
        "resource_usage": {
            "cpu_user_time_sec": round(cpu_end.user - cpu_start.user, 3),
            "cpu_system_time_sec": round(cpu_end.system - cpu_start.system, 3),
            "ram_usage_start_mb": round(mem_start / 1024 / 1024, 2),
            "ram_usage_end_mb": round(mem_end / 1024 / 1024, 2),
            "duration_sec": round(elapsed_time, 2),
            "gpu_before": gpu_before,
            "gpu_after": gpu_after
        }
    }

    return JSONResponse(content=result)
