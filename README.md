# Lenta ShelfVision

Полный прототип под кейс Lenta Tech Life Hack: React frontend + FastAPI backend.

## Что реализовано

- Загрузка видео с робота через красивый web-интерфейс.
- Асинхронная задача анализа с прогрессом.
- Backend-контракт под будущую CV/OCR нейросеть.
- Генерация отчета в CSV и XLSX.
- Полный набор колонок из задания.
- Mock/sample adapter: если имя видео совпадает с CSV организаторов, backend вернет соответствующие данные; иначе создаст демо-результат.
- Dockerfile для фронта и бэка, docker-compose для локального запуска.

## Быстрый запуск без Docker

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8765
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Открыть: http://localhost:5174

## Запуск через Docker

```bash
docker compose up --build
```

Открыть: http://localhost:5174

## Деплой

Рекомендуемая схема для хакатона: backend на Railway, frontend на Vercel. Подробные настройки лежат в `DEPLOY.md`.

## Замена mock на реальную нейросеть

Точка интеграции: `backend/app/services/mock_model.py`, функция `analyze_video(video_path, sample_dir, progress_callback)`.

Она должна вернуть `pandas.DataFrame` с колонками:

`filename, product_name, price_default, price_card, price_discount, barcode, discount_amount, id_sku, print_datetime, code, additional_info, color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max, qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr, wholesale_level_1_count, wholesale_level_1_price, wholesale_level_2_count, wholesale_level_2_price, action_price_qr, action_code_qr`.

## Фактическая архитектура фронта и бэка

### Frontend

- React + Vite SPA в `frontend/src/main.tsx`.
- Первый экран сразу является рабочим инструментом: загрузка видео, локальное video-preview, запуск анализа, прогресс пайплайна, метрики, превью первых строк результата и скачивание CSV/XLSX.
- API-адрес задается через `VITE_API_URL`, по умолчанию используется `http://localhost:8765`.

### Backend

- FastAPI-приложение в `backend/app/main.py`.
- `POST /api/jobs` принимает видео, сохраняет его в `backend/storage/uploads` и запускает обработку в `ThreadPoolExecutor`.
- `GET /api/jobs/{job_id}` возвращает статус, прогресс, метрики, ссылки на отчеты и `preview_rows` для табличного просмотра на фронте.
- `GET /api/jobs/{job_id}/download.csv` и `/download.xlsx` отдают готовые файлы из `backend/storage/reports`.
- Путь к sample-данным можно переопределить переменной окружения `LENTA_SAMPLE_DATA_DIR`; если она не задана, используется `backend/sample_data`.
- Перед sample/mock backend пробует запустить CV-детектор из `price_tag_detector`: YOLOv8n + tracking + выбор самого резкого crop. Настройки: `LENTA_USE_CV_DETECTOR`, `LENTA_DETECTOR_CPU`, `LENTA_DETECTOR_CONF`, `LENTA_DETECTOR_IOU`, `LENTA_DETECTOR_MODEL_PATH`.

### Что пока не является боевой CV/OCR-частью

- Детекция bbox уже подключена через `price_tag_detector`, если установлены зависимости и доступны веса YOLO.
- OCR текста и разбор QR пока не реализованы: для строк от CV-детектора заполняются `filename`, `frame_timestamp`, `x_min`, `y_min`, `x_max`, `y_max`, а текстовые/ценовые поля остаются пустыми.
- Если CV-детектор недоступен или не нашел объектов, backend использует sample CSV по имени видео; если совпадения по имени нет, генерируется демонстрационный результат с тем же выходным контрактом.
