import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BASE_DIR.parent
STORAGE_DIR = BASE_DIR / 'storage'
UPLOAD_DIR = STORAGE_DIR / 'uploads'
REPORT_DIR = STORAGE_DIR / 'reports'
SAMPLE_DATA_DIR = Path(os.getenv('LENTA_SAMPLE_DATA_DIR', BASE_DIR / 'sample_data')).resolve()
DETECTOR_DIR = Path(os.getenv('LENTA_DETECTOR_DIR', PROJECT_ROOT / 'price_tag_detector')).resolve()
DETECTOR_MODEL_PATH = Path(os.getenv('LENTA_DETECTOR_MODEL_PATH', DETECTOR_DIR / 'models' / 'yolov8n.pt')).resolve()
DETECTOR_ENABLED = os.getenv('LENTA_USE_CV_DETECTOR', '1').lower() not in {'0', 'false', 'no'}
DETECTOR_FORCE_CPU = os.getenv('LENTA_DETECTOR_CPU', '1').lower() not in {'0', 'false', 'no'}
DETECTOR_CONF = float(os.getenv('LENTA_DETECTOR_CONF', '0.35'))
DETECTOR_IOU = float(os.getenv('LENTA_DETECTOR_IOU', '0.45'))
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '250'))
for p in (UPLOAD_DIR, REPORT_DIR, SAMPLE_DATA_DIR):
    p.mkdir(parents=True, exist_ok=True)
