# Kamarello Frontend

React + Vite interface for uploading shelf videos, tracking job progress, previewing extracted price-tag rows, and downloading CSV/XLSX reports.

## Run

```bash
npm install
VITE_API_URL=http://localhost:8000 npm run dev
```

Production Vercel env:

```text
VITE_API_URL=https://kamarello-backend.onrender.com
```

The app uses `import.meta.env.VITE_API_URL` for every backend request.
