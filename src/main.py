from pathlib import Path
import shutil
import tempfile
from fastapi import Depends, FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
import time
import psutil

from src.analyze_metadata.analyze import check_video_metadata
from src.app.dependencies.auth import get_current_user
from src.deepfake_video_detector.detect_from_video import analyze_video
from .fake_detector.detect import check_image_fake
from src.app.routes import auth

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["auth"])


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Support Chat API",
        version="1.0.0",
        description="API for admin/operator/widget with JWT",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if not path.startswith("/widget"):
                openapi_schema["paths"][path][method]["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi




@app.post("/check-image/")
async def check_image(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    AI tomonidan yaratilgan rasm (fake) yoki yo‘qligini va watermark mavjudligini tekshiradi.

    - Rasmni vaqtincha (`tempfile`) saqlaydi (doimiy saqlash yo‘q)
    - AI generated checker orqali real yoki fake ekanligini tekshiradi
    - Watermark checker orqali suv belgisi borligini aniqlaydi
    - CPU, RAM, va vaqtni o‘lchab beradi

    Args:
        file (UploadFile): Yuklangan rasm

    Returns:
        dict: AI natijasi, watermark natijasi va texnik metrikalar
    """
    process = psutil.Process()
    start_time = time.time()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Faylni RAMda o‘qib olish
    content = await file.read()

    # AI generated checker — faylsiz ishlaydi
    ai_result = check_image_fake(content)

    duration = time.time() - start_time
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    cpu_percent = process.cpu_percent(interval=0.1)

    return {
        "ai_generated_checker": ai_result,
        "metrics": {
            "duration_seconds": round(duration, 3),
            "memory_used_MB": round(mem_after - mem_before, 2),
            "cpu_percent": cpu_percent,
        }
    }



@app.post("/check-video/")
async def analyze_uploaded_video(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    Video faylni quyidagi jihatlar bo‘yicha tekshiradi:
    - AI generated video (deepfake) aniqlovchi model orqali
    - Watermark bor-yo‘qligi
    - Metadata (recorded/encoded date, encoder) tahlili

    Shuningdek:
    - CPU va RAM statistikasi
    - GPU ishlatilish statistikasi
    - Har bir tekshiruv uchun vaqt

    Args:
        file (UploadFile): Yuklangan video fayl

    Returns:
        JSONResponse: Tahlil natijalari, texnik metrikalar va xatolik yuz bersa — xatolik haqida xabar.
    """
    tmp_file_path = None

    try:
        suffix = Path(file.filename).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_file_path = Path(tmp.name)

        # Monitoring: boshlanishi
        process = psutil.Process()
        start_time = time.time()
        cpu_start = process.cpu_times()
        mem_start = process.memory_info().rss

        # Asosiy tekshiruvlar
        metadata_video = check_video_metadata(str(tmp_file_path))
        ai_generated_check = analyze_video(video_path=str(tmp_file_path))

        # Monitoring: yakuni
        elapsed_time = time.time() - start_time
        cpu_end = process.cpu_times()
        mem_end = process.memory_info().rss

        result = {
            "ai_generated_checker": ai_generated_check,
            "metadata_checker": metadata_video,
            "resource_usage": {
                "cpu_user_time_sec": round(cpu_end.user - cpu_start.user, 3),
                "cpu_system_time_sec": round(cpu_end.system - cpu_start.system, 3),
                "ram_usage_start_mb": round(mem_start / 1024 / 1024, 2),
                "ram_usage_end_mb": round(mem_end / 1024 / 1024, 2),
                "duration_sec": round(elapsed_time, 2),
            }
        }

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

    finally:
        # Faylni tozalash
        if tmp_file_path and tmp_file_path.exists():
            tmp_file_path.unlink()
