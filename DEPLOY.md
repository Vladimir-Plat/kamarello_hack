# Deploy: Railway backend + Vercel frontend

## Backend on Railway

Create a new Railway service from GitHub:

- Repository: `Oleg4311/kamarello_hack`
- Root Directory: `backend`
- Builder: Dockerfile
- Healthcheck Path: `/health`

Environment variables:

```env
LENTA_USE_CV_DETECTOR=1
LENTA_DETECTOR_CPU=1
LENTA_DETECTOR_CONF=0.35
LENTA_DETECTOR_IOU=0.45
MAX_UPLOAD_MB=250
```

If the Railway plan cannot handle YOLO/OpenCV memory usage, set:

```env
LENTA_USE_CV_DETECTOR=0
```

For persistent uploads/reports, attach a Railway volume mounted at:

```text
/app/storage
```

## Frontend on Vercel

Create a new Vercel project from the same GitHub repository:

- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: `dist`

Environment variable:

```env
VITE_API_URL=https://<railway-backend-domain>
```

Redeploy frontend after changing `VITE_API_URL`, because Vite embeds this variable during build.
