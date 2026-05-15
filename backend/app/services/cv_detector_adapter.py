from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from app.config import DETECTOR_CONF, DETECTOR_ENABLED, DETECTOR_FORCE_CPU, DETECTOR_IOU, DETECTOR_MODEL_PATH, REPORT_DIR

REQUIRED_COLUMNS = [
    'filename','product_name','price_default','price_card','price_discount','barcode','discount_amount','id_sku',
    'print_datetime','code','additional_info','color','special_symbols','frame_timestamp','x_min','y_min','x_max','y_max',
    'qr_code_barcode','price1_qr','price2_qr','price3_qr','price4_qr','wholesale_level_1_count','wholesale_level_1_price',
    'wholesale_level_2_count','wholesale_level_2_price','action_price_qr','action_code_qr'
]


def _load_processor_class():
    from app.services.price_tag_detector_core import PriceTagProcessor
    return PriceTagProcessor


def _records_to_result_frame(video_path: Path, records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        x = int(record.get('x', 0) or 0)
        y = int(record.get('y', 0) or 0)
        w = int(record.get('w', 0) or 0)
        h = int(record.get('h', 0) or 0)
        rows.append({
            'filename': video_path.name,
            'product_name': '',
            'price_default': '',
            'price_card': '',
            'price_discount': '',
            'barcode': '',
            'discount_amount': '',
            'id_sku': '',
            'print_datetime': '',
            'code': '',
            'additional_info': (
                f"cv_detector crop={record.get('crop_file', '')}; "
                f"track_id={record.get('tracker_id', '')}; "
                f"sharpness={record.get('sharpness', '')}"
            ),
            'color': '',
            'special_symbols': '',
            'frame_timestamp': int(record.get('timestamp_ms', 0) or 0),
            'x_min': x,
            'y_min': y,
            'x_max': x + w,
            'y_max': y + h,
            'qr_code_barcode': '',
            'price1_qr': '',
            'price2_qr': '',
            'price3_qr': '',
            'price4_qr': '',
            'wholesale_level_1_count': '',
            'wholesale_level_1_price': '',
            'wholesale_level_2_count': '',
            'wholesale_level_2_price': '',
            'action_price_qr': '',
            'action_code_qr': '',
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def try_analyze_with_cv_detector(
    video_path: Path,
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame | None:
    if not DETECTOR_ENABLED:
        return None

    if progress_callback:
        progress_callback(5)

    output_dir = REPORT_DIR / f'detector_{video_path.stem}'
    model_path = str(DETECTOR_MODEL_PATH) if DETECTOR_MODEL_PATH.exists() else 'yolov8n.pt'
    processor_cls = _load_processor_class()
    processor = processor_cls(
        model_path=model_path,
        output_dir=str(output_dir),
        conf_threshold=DETECTOR_CONF,
        iou_threshold=DETECTOR_IOU,
        use_gpu=not DETECTOR_FORCE_CPU,
    )

    if progress_callback:
        progress_callback(15)
    records = processor.process_video(str(video_path))
    if progress_callback:
        progress_callback(90)

    if not records:
        return None
    return _records_to_result_frame(video_path, records)
