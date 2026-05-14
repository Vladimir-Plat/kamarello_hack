from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
STORAGE_DIR = BASE_DIR / 'storage'
UPLOAD_DIR = STORAGE_DIR / 'uploads'
REPORT_DIR = STORAGE_DIR / 'reports'
SAMPLE_DATA_DIR = BASE_DIR / 'sample_data'
for p in (UPLOAD_DIR, REPORT_DIR, SAMPLE_DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
