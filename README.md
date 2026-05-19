# Kamarello Hack

Prototype for Lenta Tech Life Hack: Vercel React frontend uploads shelf videos, Render FastAPI backend detects price tags with YOLOv8, runs OCR/parsing on the best crops, and exports CSV/XLSX.

## Architecture

Current deployed scheme:

`Vercel frontend -> Render backend -> YOLO/OCR processing -> CSV/XLSX download`

Frontend: https://kamarello-hack.vercel.app

Backend: https://kamarello-backend.onrender.com

Vercel only serves the React app. It does not run YOLO, OCR, or LLM code. All heavy video processing runs on the Render backend, and the frontend talks to it through HTTPS API calls.

## API

- `GET /` returns backend status and links.
- `GET /api/health` returns `{"status":"ok","service":"kamarello-backend"}`.
- `POST /api/jobs` accepts multipart video upload as `file`.
- `GET /api/jobs/{job_id}` returns status, progress, metrics, preview rows, and report URLs.
- `GET /api/jobs/{job_id}/download.csv` downloads CSV.
- `GET /api/jobs/{job_id}/download.xlsx` downloads XLSX.

## Processing Pipeline

The primary path is computer vision plus OCR, not VLM:

1. YOLOv8 detects price tags in video frames.
2. Tracking keeps the sharpest crop for each detected tag.
3. PaddleOCR reads text from crop variants.
4. Rule-based parser extracts product, prices, discounts, barcode, SKU, color, and promo symbols.
5. Optional LLM refiner can improve fields using OCR text only, never images.

If OCR is disabled or unavailable, the backend does not crash. It still returns bbox/crop metadata and falls back to sample/mock output when needed.

## Local Run

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Optional OCR dependencies:

```bash
cd backend
pip install -r requirements-ocr.txt
```

Frontend:

```bash
cd frontend
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

## Deployment

### Frontend: Vercel

Project URL: https://kamarello-hack.vercel.app

Set this environment variable in Vercel:

```text
VITE_API_URL=https://kamarello-backend.onrender.com
```

Vercel path:

`Project Settings -> Environment Variables -> add VITE_API_URL -> Redeploy`

Vite embeds `VITE_API_URL` during build, so redeploy is required after changing it.

### Backend: Render

Backend public URL: https://kamarello-backend.onrender.com

Use the backend Docker service. Recommended Render environment:

```text
LENTA_USE_CV_DETECTOR=1
LENTA_DETECTOR_CPU=1
LENTA_USE_OCR=1
LENTA_USE_LLM_REFINER=0
LENTA_LLM_MODEL_PATH=
LENTA_OCR_LANG=ru
LENTA_MAX_CROPS_PER_VIDEO=300
```

If PaddleOCR does not fit into the free Render instance or crashes because of memory, set:

```text
LENTA_USE_OCR=0
```

The backend will keep running and will return detector metadata plus fallback fields.

Optional LLM mode:

```text
LENTA_USE_LLM_REFINER=1
LENTA_LLM_MODEL_PATH=/path/to/qwen2.5-3b-instruct-q4_k_m.gguf
```

Use a small quantized text model such as Qwen2.5-3B-Instruct Q4_K_M. Do not use 7B on free Render.

## Frontend API URL

Frontend uses one API helper:

```ts
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
```

All requests go through it:

- `POST ${API_BASE_URL}/api/jobs`
- `GET ${API_BASE_URL}/api/jobs/${jobId}`
- `GET ${API_BASE_URL}/api/jobs/${jobId}/download.csv`
- `GET ${API_BASE_URL}/api/jobs/${jobId}/download.xlsx`

## Verification

After deployment check:

1. https://kamarello-backend.onrender.com/api/health opens.
2. https://kamarello-backend.onrender.com/docs opens.
3. Frontend posts videos to `https://kamarello-backend.onrender.com/api/jobs`, not localhost.
4. Browser console has no CORS errors.
5. CSV/XLSX downloads from the Render backend.
6. A sleeping free Render instance may need 50+ seconds for the first request.
