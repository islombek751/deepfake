# 🚀 FastAPI Project

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
    .
    ├── requirements
    ├── compose
        ├── Dockerfile.gpu    ← bu GPU mavjud bolsa ishlaydi.
        └── Dockerfile.cpu
    ├── local.yml
    ├── README.md
    └── src/
        ├── main.py    ← bu yerda API endpointlar yozilgan
        ├── analyze_metadata/
        ├── deepfake_video_detector/
        │   └── models/  ← bu yerga Deepfake video modeli joylashtiriladi
        ├── watermark_detector/
        │   └── models/  ← bu yerga Watermark modeli joylashtiriladi
        └── fake_detector/
            └── models/  ← bu yerga AI-generated modeli joylashtiriladi


5. ✨ Eslatma
Model fayllari .gitignore orqali git’da kuzatilmaydi. Ularni alohida yuklab olish majburiy.

media-checker-cpu — CPU bilan ishlaydigan servis (hamma kompyuterda ish beradi)

media-checker-gpu — faqat NVIDIA GPU va nvidia-docker mavjud bo‘lsa ishlaydi


6. Server uchun minimal va tavsiya qilingan harakteristika:

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
