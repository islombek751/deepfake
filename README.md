# 🚀 FastAPI Project

## ⚙️ O‘rnatish

1. 🔽 Repository’ni klonlash:
   ```bash
   git clone https://github.com/yourusername/your-fastapi-project.git
   cd your-fastapi-project

2. 🔽 Virtual environment yaratish (ixtiyoriy, tavsiya qilinadi):
    ```bash
    python -m venv venv
    source venv/bin/activate    # Linux/Mac
    .\venv\Scripts\activate  

3. 🔽 Kutubxonalarni o‘rnatish:
    ```bash
    pip install -r requirements.txt

3. 🔽 Modellarni kerakli papklarga yuklab olish:
    Watermark Detect funksiyasi ishlashi uchun quyidagi linkdagi modelni yuklab oling:

    https://disk.yandex.com/d/8ylxsLs1uhj-Xg

    Va src/watermark_detector/models/ papkasiga ko'chirib o'tkazing.

    ___________________________________________________________________________________

    Ai-generated Detector funksiyasi ishlashi uchun quyidagi linkdagi modelni yuklab oling:

    https://disk.yandex.com/d/OlxjNLvxb01xpg

    Va src/fake_detector/models/ papkasiga ko'chirib o'tkazing.

🚀 Loyihani ishga tushirish
    ```bash
    uvicorn main:app --reload


📁 Loyiha tuzilmasi
```bash
    .
    ├── main.py
    ├── requirements.txt
    ├── README.md
    └── src/
        ├── watermark_detector/
        │   └── models/  ← bu yerga Watermark modeli joylashtiriladi
        └── fake_detector/
            └── models/  ← bu yerga AI-generated modeli joylashtiriladi


✨ Eslatma
Model fayllari .gitignore orqali git’da kuzatilmaydi. Ularni alohida yuklab olish majburiy.