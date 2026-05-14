from __future__ import annotations

import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import UPLOAD_DIR, REPORT_DIR, SAMPLE_DATA_DIR
from app.schemas import JobCreateResponse, JobStatusResponse
from app.services.mock_model import analyze_video

app = FastAPI(title='Lenta ShelfVision API', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

executor = ThreadPoolExecutor(max_workers=2)
JOBS: Dict[str, Dict[str, Any]] = {}


def _set_progress(job_id: str, progress: int):
    JOBS[job_id]['progress'] = progress


def _run_analysis(job_id: str):
    job = JOBS[job_id]
    try:
        job['status'] = 'processing'
        df = analyze_video(Path(job['upload_path']), SAMPLE_DATA_DIR, lambda p: _set_progress(job_id, p))
        csv_path = REPORT_DIR / f'{job_id}.csv'
        xlsx_path = REPORT_DIR / f'{job_id}.xlsx'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='price_tags')
            ws = writer.book['price_tags']
            ws.freeze_panes = 'A2'
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            for column in ws.columns:
                letter = column[0].column_letter
                max_len = max(len(str(c.value)) if c.value is not None else 0 for c in column[:60])
                ws.column_dimensions[letter].width = min(max(max_len + 2, 12), 42)
        job.update({
            'status': 'done',
            'progress': 100,
            'rows_count': len(df),
            'csv_path': str(csv_path),
            'xlsx_path': str(xlsx_path),
            'metrics': {
                'price_tags': int(len(df)),
                'unique_barcodes': int(df['barcode'].astype(str).replace('', pd.NA).nunique()),
                'avg_confidence': 0.87,
                'mode': 'mock-or-sample-adapter',
            },
        })
    except Exception as exc:
        job.update({'status': 'failed', 'error': str(exc), 'progress': 100})


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/api/jobs', response_model=JobCreateResponse)
async def create_job(file: UploadFile = File(...)):
    allowed = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail='Загрузите видеофайл: mp4, mov, avi, mkv или webm')
    job_id = str(uuid.uuid4())
    safe_name = Path(file.filename or f'video{suffix}').name
    upload_path = UPLOAD_DIR / f'{job_id}_{safe_name}'
    with upload_path.open('wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    JOBS[job_id] = {
        'job_id': job_id,
        'status': 'queued',
        'progress': 0,
        'filename': safe_name,
        'upload_path': str(upload_path),
        'rows_count': 0,
        'metrics': {},
    }
    executor.submit(_run_analysis, job_id)
    return JobCreateResponse(job_id=job_id, status='queued', message='Видео принято, анализ запущен')


@app.get('/api/jobs/{job_id}', response_model=JobStatusResponse)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='Задача не найдена')
    base = f'/api/jobs/{job_id}'
    return JobStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job['progress'],
        filename=job['filename'],
        rows_count=job.get('rows_count', 0),
        metrics=job.get('metrics', {}),
        error=job.get('error'),
        csv_url=f'{base}/download.csv' if job['status'] == 'done' else None,
        xlsx_url=f'{base}/download.xlsx' if job['status'] == 'done' else None,
    )


@app.get('/api/jobs/{job_id}/download.csv')
def download_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get('status') != 'done':
        raise HTTPException(status_code=404, detail='Отчет еще не готов')
    return FileResponse(job['csv_path'], filename=f'lenta_price_tags_{job_id}.csv', media_type='text/csv')


@app.get('/api/jobs/{job_id}/download.xlsx')
def download_xlsx(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get('status') != 'done':
        raise HTTPException(status_code=404, detail='Отчет еще не готов')
    return FileResponse(job['xlsx_path'], filename=f'lenta_price_tags_{job_id}.xlsx', media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
