# Deploy: Render Backend + Vercel Frontend

## Backend on Render

Create a Render Docker service from GitHub:

- Root Directory: `backend`
- Builder: Dockerfile
- Healthcheck Path: `/api/health`
- Public URL: `https://kamarello-backend.onrender.com`

Environment variables:

```env
LENTA_USE_CV_DETECTOR=1
LENTA_DETECTOR_CPU=1
LENTA_DETECTOR_CONF=0.35
LENTA_DETECTOR_IOU=0.45
LENTA_USE_OCR=1
LENTA_USE_LLM_REFINER=0
LENTA_LLM_MODEL_PATH=
LENTA_OCR_LANG=ru
LENTA_MAX_CROPS_PER_VIDEO=300
MAX_UPLOAD_MB=250
```

If the free instance cannot handle PaddleOCR memory usage:

```env
LENTA_USE_OCR=0
```

## Frontend on Vercel

Vercel project:

- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`
- Public URL: `https://kamarello-hack.vercel.app`

Environment variable:

```env
VITE_API_URL=https://kamarello-backend.onrender.com
```

Redeploy frontend after changing `VITE_API_URL`, because Vite embeds this variable during build.
