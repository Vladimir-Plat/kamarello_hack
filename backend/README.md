# Lenta ShelfVision API

FastAPI backend для хакатона Lenta Tech: прием видео робота, запуск анализа, выдача CSV/XLSX отчета.

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API: http://localhost:8000/docs

## Контракт

- `POST /api/jobs` — multipart upload `file`, возвращает `job_id`.
- `GET /api/jobs/{job_id}` — статус, прогресс, метрики, ссылки на отчеты.
- `GET /api/jobs/{job_id}/download.csv` — CSV.
- `GET /api/jobs/{job_id}/download.xlsx` — Excel.

## Где подключать нейросеть

Файл `app/services/mock_model.py`, функция `analyze_video(...)`. Сейчас она берет CSV из `sample_data`, если имя видео совпадает, иначе генерирует демо-результат. Когда будет готов CV/OCR пайплайн, нужно заменить только эту функцию, сохранив список выходных колонок.
