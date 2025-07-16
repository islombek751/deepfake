# 🚀 FastAPI Project

# ⚙️ Server uchun minimal va tavsiya qilingan harakteristika:

    | Resurs             | Minimal      | Tavsiya (GPU bo‘lsa)             |
    | ------------------ | ------------ | -------------------------------- |
    | CPU                | 4 cores      | 8 cores                          |
    | GPU                | ❌ yo‘q       | ✅ NVIDIA RTX 3060 yoki Tesla T4  |
    | CUDA               | ❌ kerak emas | ✅ CUDA 12.1                      |
    | RAM                | 8 GB         | 16 GB+                           |
    | Disk               | 10 GB SSD    | 20 GB SSD (model fayllari uchun) |
    | OS                 | Ubuntu 22.04 | Ubuntu 22.04                     |
    | Python             | 3.10         | 3.10                             |
    | PostgreSQL         | 14+          | 14+                              |
    | Gunicorn + Uvicorn | ishlatiladi  | ishlatiladi                      |



## ⚙️ O‘rnatish

1. 🔽 Repository’ni klonlash:
   ```bash
   git clone https://git.uzinfocom.uz/agrod/ministry-of-water-resources/media-checker/backend.git
   cd backend


2. 🔽 Modellarni kerakli papklarga yuklab olish:
    
    Watermark Detect funksiyasi ishlashi uchun quyidagi linkdagi modelni yuklab oling:
    
    https://disk.yandex.com/d/8ylxsLs1uhj-Xg

    Va src/watermark_detector/models/ papkasiga ko'chirib o'tkazing.

    ___________________________________________________________________________________

    Ai-generated Detector funksiyasi ishlashi uchun quyidagi linkdagi modelni yuklab oling:
    
    https://disk.yandex.com/d/OlxjNLvxb01xpg


    Va src/fake_detector/models/ papkasiga ko'chirib o'tkazing.

    ____________________________________________________________________________________

    DeepFake video detector funksiyasi ishlashi uchun quyidagi linkdagi modelni yuklab oling:

    https://disk.yandex.com/d/8QB3RJjWzN4keQ

    Va src/deepfake_video_detector/models/ papkasiga ko'chirib o'tkazing.


3. 🚀 Loyihani ishga tushirish
    
    ```bash
    #agar sizda GPU mavjud bo'lmasa

    docker compose -f local.yml up --build media-checker-cpu

    http://127.0.0.1:8000/docs

    YOKI

    #agar sizda GPU mavjud bo'lsa

    docker compose -f local.yml up --build media-checker-gpu


    http://127.0.0.1:8001/docs

4. 📁 Loyiha tuzilmasi
    ```bash
        ├── .env                         # Muhit o'zgaruvchilari
        ├── alembic.ini                  # Alembic sozlamalari
        ├── local.yml                    # Docker Compose config
        ├── README.md
        │
        ├── alembic/                     # Alembic migratsiya fayllari
        │   ├── versions/                # Har bir migratsiya fayli shu yerda
        │   ├── env.py                   # Alembic konfiguratsiyasi
        │
        ├── compose/                     # Dockerfile.lar
        │   ├── Dockerfile.cpu
        │   └── Dockerfile.gpu
        │
        ├── requirements/               # Turli muhitlar uchun talablar
        │   ├── base.txt
        │   ├── requirements.cpu.txt
        │   └── requirements.gpu.txt
        │
        ├── src/
        │   ├── main.py                  # FastAPI kirish nuqtasi
        │   ├── config.py                # .env faylni o‘qish
        │   ├── __init__.py
        │   │
        │   ├── app/                     # Asosiy backend logika
        │   │   ├── deps.py              # General depends
        │   │   ├── core/                # Auth + JWT funksiyalari
        │   │   │   ├── jwt.py
        │   │   │   └── security.py
        │   │   ├── db/                  # SQLAlchemy bazaviy sozlamalar
        │   │   │   ├── base.py
        │   │   │   └── session.py
        │   │   ├── dependencies/        # FastAPI depends (auth uchun)
        │   │   │   └── auth.py
        │   │   ├── models/              # SQLAlchemy ORM modellar
        │   │   │   └── user.py
        │   │   ├── schemas/             # Pydantic sxemalar
        │   │   │   └── user.py
        │   │   └── routes/              # API endpointlar
        │   │       └── auth.py
        │
        │   ├── analyze_metadata/        # Video metadatasini tahlil qilish
        │   │   └── analyze.py
        │
        │   ├── deepfake_video_detector/ # Deepfake video aniqlash
        │   │   ├── models/              # PyTorch model fayli (.pth)
        │   │   ├── detect_from_video.py
        │   │   ├── models.py
        │   │   ├── transform.py
        │   │   └── xception.py
        │
        │   ├── fake_detector/           # Rasm bo‘yicha AI aniqlovchi modul
        │   │   ├── models/
        │   │   │   └── model_epoch_24.pth
        │   │   ├── detect.py
        │   │   ├── model.py
        │   │   └── custom_dataset.py
        │
        │   └── watermark_detector/      # Watermark tekshiruvchi modul
        │       ├── models/
        │       │   └── convnext-tiny_watermarks_detector.pth
        │       ├── predictor.py
        │       ├── utils.py
        │       └── watermark_detector.py



5. ✨ Eslatma
Model fayllari .gitignore orqali git’da kuzatilmaydi. Ularni alohida yuklab olish majburiy.

media-checker-cpu — CPU bilan ishlaydigan servis (hamma kompyuterda ish beradi)

media-checker-gpu — faqat NVIDIA GPU va nvidia-docker mavjud bo‘lsa ishlaydi
