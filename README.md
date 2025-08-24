# Deepfake Detection with FastAPI

Ushbu loyiha video fayllarini tahlil qilish va ularning haqiqiyligini aniqlash uchun **Deepfake aniqlash tizimini** yaratadi. Loyiha **FastAPI** yordamida qurilgan bo‘lib, foydalanuvchilarga video fayllarni yuklash va ularning haqiqiyligini tekshirish imkonini beradi.

---

## 📁 Loyiha tuzilmasi

├── alembic/ # Ma'lumotlar bazasi migratsiyalari
├── src/
│ ├── analyze_metadata/
│ │ └── analyze.py # Video metadata tahlil qilish
│ ├── app/
│ │ ├── core/ # Ilovaning asosiy konfiguratsiyasi
│ │ ├── db/ # Ma'lumotlar bazasi modullari
│ │ ├── dependencies/ # FastAPI dependency modul
│ │ ├── models/ # SQLAlchemy modellari
│ │ ├── routes/ # API endpointlar
│ │ ├── schemas/ # Pydantic schemas
│ │ └── deps.py
│ ├── deepfake_video_detector/
│ │ └── detect_from_video.py # Video fayldan Deepfake aniqlash
│ └── fake_detector/
│ ├── init.py
│ └── detect.py # Asosiy detektor moduli
├── config.py # Ilova konfiguratsiyasi
├── main.py # FastAPI ilovasi
├── .env # Atrof-muhit o'zgaruvchilari
├── .gitignore
├── alembic.ini
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

## 🛠️ Talablar

- [ffmpeg](https://ffmpeg.org/download.html)

---

## 🚀 Ishga tushirish

### 1. Muhitni tayyorlash:

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
2. Kerakli kutubxonalarni o‘rnatish:
bash
Copy
Edit
pip install -r requirements.txt
3. Ma'lumotlar bazasini yaratish va migratsiyalarni qo‘llash:
bash
Copy
Edit
alembic upgrade head
4. Ilovani ishga tushirish:
bash
Copy
Edit
uvicorn src.app.main:app --reload
Ilova http://127.0.0.1:8000 manzilida ishga tushadi.
Swagger interfeysi http://127.0.0.1:8000/docs/ manzilida.

📄 API Endpoints
POST /check-image/ : Rasm yuborib, uning haqiqiyligini aniqlash

POST /check-video/ : Video faylni yuborib, uning haqiqiyligini aniqlash

