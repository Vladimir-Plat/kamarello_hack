from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import MAX_UPLOAD_MB, REPORT_DIR, SAMPLE_DATA_DIR, UPLOAD_DIR
from app.schemas import JobCreateResponse, JobHistoryResponse, JobStatusResponse
from app.services.mock_model import analyze_video
from app.storage_service import load_jobs, save_jobs

app = FastAPI(title="Lenta ShelfVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kamarello-hack.vercel.app",
        "https://kamarello-hack-git-main-olegs-projects-4acc0c1b.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=2)
JOBS: Dict[str, Dict[str, Any]] = load_jobs()
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _persist() -> None:
    save_jobs(JOBS)


def _set_progress(job_id: str, progress: int):
    job = JOBS.get(job_id)
    if not job:
        return

    job["progress"] = progress
    job["updated_at"] = _now_iso()
    _persist()


def _run_analysis(job_id: str):
    job = JOBS[job_id]

    try:
        job["status"] = "processing"
        job["updated_at"] = _now_iso()
        _persist()

        df = analyze_video(
            Path(job["upload_path"]),
            SAMPLE_DATA_DIR,
            lambda p: _set_progress(job_id, p),
        )

        preview_df = df.head(20).astype(object).where(pd.notna(df.head(20)), None)

        csv_path = REPORT_DIR / f"{job_id}.csv"
        xlsx_path = REPORT_DIR / f"{job_id}.xlsx"

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="price_tags")
            ws = writer.book["price_tags"]
            ws.freeze_panes = "A2"

            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)

            for column in ws.columns:
                letter = column[0].column_letter
                max_len = max(
                    len(str(c.value)) if c.value is not None else 0
                    for c in column[:60]
                )
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 42)

        job.update(
            {
                "status": "done",
                "progress": 100,
                "rows_count": len(df),
                "csv_path": str(csv_path),
                "xlsx_path": str(xlsx_path),
                "preview_rows": preview_df.to_dict(orient="records"),
                "metrics": {
                    "price_tags": int(len(df)),
                    "unique_barcodes": int(
                        df["barcode"].astype(str).replace("", pd.NA).nunique()
                    ),
                    "avg_confidence": 0.87,
                    "mode": df.attrs.get("mode", "mock-or-sample-adapter"),
                },
                "updated_at": _now_iso(),
                "finished_at": _now_iso(),
            }
        )
        _persist()

    except Exception as exc:
        job.update(
            {
                "status": "failed",
                "error": str(exc),
                "progress": 100,
                "updated_at": _now_iso(),
                "finished_at": _now_iso(),
            }
        )
        _persist()


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs", "health": "/api/health"}


@app.get("/api/health")
def api_health():
    return {"status": "ok", "service": "kamarello-backend"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "kamarello-backend"}


@app.post("/api/jobs", response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)):
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Загрузите видеофайл: mp4, mov, avi, mkv или webm",
        )

    job_id = str(uuid.uuid4())
    safe_name = Path(file.filename or f"video{suffix}").name
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"

    written = 0

    with upload_path.open("wb") as buffer:
        while True:
            chunk = file.file.read(1024 * 1024)

            if not chunk:
                break

            written += len(chunk)

            if written > MAX_UPLOAD_BYTES:
                upload_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"Файл слишком большой. Лимит: {MAX_UPLOAD_MB} MB",
                )

            buffer.write(chunk)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "filename": safe_name,
        "upload_path": str(upload_path),
        "rows_count": 0,
        "metrics": {},
        "preview_rows": [],
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    _persist()
    executor.submit(_run_analysis, job_id)

    return JobCreateResponse(
        job_id=job_id,
        status="queued",
        message="Видео принято, анализ запущен",
    )


@app.get("/api/history", response_model=list[JobHistoryResponse])
def get_history():
    jobs = sorted(
        JOBS.values(),
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )

    result = []

    for job in jobs:
        job_id = job["job_id"]
        is_done = job.get("status") == "done"

        result.append(
            JobHistoryResponse(
                job_id=job_id,
                status=job.get("status", "queued"),
                progress=job.get("progress", 0),
                filename=job.get("filename", "unknown"),
                rows_count=job.get("rows_count", 0),
                created_at=job.get("created_at"),
                updated_at=job.get("updated_at"),
                error=job.get("error"),
                csv_url=f"/api/jobs/{job_id}/download.csv" if is_done else None,
                xlsx_url=f"/api/jobs/{job_id}/download.xlsx" if is_done else None,
            )
        )

    return result


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    base = f"/api/jobs/{job_id}"

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        filename=job["filename"],
        rows_count=job.get("rows_count", 0),
        metrics=job.get("metrics", {}),
        preview_rows=job.get("preview_rows", []),
        error=job.get("error"),
        created_at=job.get("created_at"),
        updated_at=job.get("updated_at"),
        csv_url=f"{base}/download.csv" if job["status"] == "done" else None,
        xlsx_url=f"{base}/download.xlsx" if job["status"] == "done" else None,
    )


@app.get("/api/jobs/{job_id}/download.csv")
def download_csv(job_id: str):
    job = JOBS.get(job_id)

    if not job or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Отчет еще не готов")

    return FileResponse(
        job["csv_path"],
        filename=f"lenta_price_tags_{job_id}.csv",
        media_type="text/csv",
    )


@app.get("/api/jobs/{job_id}/download.xlsx")
def download_xlsx(job_id: str):
    job = JOBS.get(job_id)

    if not job or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Отчет еще не готов")

    return FileResponse(
        job["xlsx_path"],
        filename=f"lenta_price_tags_{job_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
