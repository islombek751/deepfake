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
    #agar sizda GPU mavjud bolmasa

    docker compose -f local.yml up --build media-checker-cpu


    YOKI

    #agar sizda GPU mavjud bolsa

    docker compose -f local.yml up --build media-checker-gpu


    http://127.0.0.1:8000/docs

4. 📁 Loyiha tuzilmasi
    ```bash
    .
    ├── requirements
    ├── compose
        ├── Dockerfile.gpu    ← bu GPU mavjud bolsa ishlaydi.
        └── Dockerfile.cpu
    ├── local.yml
    ├── README.md
    ├── uploaded_images
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