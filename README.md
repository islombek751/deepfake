Deepfake Detection with FastAPI

Ushbu loyiha video fayllarini tahlil qilish va ularning haqiqiyligini aniqlash uchun Deepfake aniqlash tizimini yaratadi. Loyiha FastAPI yordamida qurilgan bo'lib, foydalanuvchilarga video fayllarni yuklash va ularning haqiqiyligini tekshirish imkonini beradi.

📁 Loyiha tuzilmasi

├── alembic/                     # Ma'lumotlar bazasi migratsiyalari
├── src/
│   ├── analyze_metadata/
│   │   └── analyze.py           # Video metadata tahlil qilish
│   ├── app/
│   │   ├── core/                # Ilovaning asosiy konfiguratsiyasi
│   │   ├── db/                  # Ma'lumotlar bazasi modullari
│   │   ├── dependencies/        # FastAPI dependency modul
│   │   ├── models/              # SQLAlchemy modellari
│   │   ├── routes/              # API endpointlar
│   │   ├── schemas/             # Pydantic schemas
│   │   └── deps.py
│   ├── deepfake_video_detector/
│   │   └── detect_from_video.py # Video fayldan Deepfake aniqlash
│   └── fake_detector/
│       ├── __init__.py
│       ├── detect.py            # Asosiy detektor moduli
│       └── __init__.py
├── config.py                     # Ilova konfiguratsiyasi
├── main.py                       # FastAPI ilovasi
├── .env                          # Atrof-muhit o'zgaruvchilari
├── .gitignore
├── alembic.ini
├── requirements.txt
└── README.md


🛠️ Talablar

ffmpeg

🚀 Ishga tushirish

Muhitni tayyorlash:

python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Kerakli kutubxonalarni o'rnatish:

pip install -r requirements.txt


Ma'lumotlar bazasini yaratish va migratsiyalarni qo'llash:

alembic upgrade head


Ilovani ishga tushirish:

uvicorn src.app.main:app --reload


Ilova http://127.0.0.1:8000 manzilida ishga tushadi.
Swagger http://127.0.0.1:8000/docs/ manzilida.

📄 API Endpoints

POST /check-image/: Rasm yuborib, uning haqiqiyligini aniqlash

POST /check-video/: Video faylni yuborib, uning haqiqiyligini aniqlash.