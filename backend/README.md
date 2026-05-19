# Kamarello Backend

FastAPI backend for video upload, YOLOv8 price-tag detection, optional PaddleOCR, rule-based parsing, and CSV/XLSX export.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

Optional OCR:

```bash
pip install -r requirements-ocr.txt
```

## Endpoints

- `GET /`
- `GET /api/health`
- `POST /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/download.csv`
- `GET /api/jobs/{job_id}/download.xlsx`

## Render Env

```text
LENTA_USE_CV_DETECTOR=1
LENTA_DETECTOR_CPU=1
LENTA_USE_OCR=1
LENTA_USE_LLM_REFINER=0
LENTA_LLM_MODEL_PATH=
LENTA_OCR_LANG=ru
LENTA_MAX_CROPS_PER_VIDEO=300
```

Set `LENTA_USE_OCR=0` if PaddleOCR is too heavy for the instance.
