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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Открыть: http://localhost:5173

## Запуск через Docker

```bash
docker compose up --build
```

Открыть: http://localhost:5173

## Замена mock на реальную нейросеть

Точка интеграции: `backend/app/services/mock_model.py`, функция `analyze_video(video_path, sample_dir, progress_callback)`.

Она должна вернуть `pandas.DataFrame` с колонками:

`filename, product_name, price_default, price_card, price_discount, barcode, discount_amount, id_sku, print_datetime, code, additional_info, color, special_symbols, frame_timestamp, x_min, y_min, x_max, y_max, qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr, wholesale_level_1_count, wholesale_level_1_price, wholesale_level_2_count, wholesale_level_2_price, action_price_qr, action_code_qr`.
