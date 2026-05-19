from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from app.config import (
    DETECTOR_CONF,
    DETECTOR_ENABLED,
    DETECTOR_FORCE_CPU,
    DETECTOR_IOU,
    DETECTOR_MODEL_PATH,
    LLM_REFINER_ENABLED,
    MAX_CROPS_PER_VIDEO,
    REPORT_DIR,
)
from app.services.barcode_service import read_codes_from_image
from app.services.llm_text_refiner import refine_with_llm
from app.services.ocr_service import extract_ocr_payload
from app.services.price_tag_parser import parse_price_tag_ocr

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
    for index, record in enumerate(records[:MAX_CROPS_PER_VIDEO]):
        x = int(record.get('x', 0) or 0)
        y = int(record.get('y', 0) or 0)
        w = int(record.get('w', 0) or 0)
        h = int(record.get('h', 0) or 0)
        crop_name = str(record.get('crop_file', '') or '')
        crop_path = REPORT_DIR / f'detector_{video_path.stem}' / 'crops' / crop_name

        ocr_result = extract_ocr_payload(crop_path) if crop_name else {
            "raw_lines": [], "merged_rows": [], "markdown_table": "", "plain_text": ""
        }
        parsed = parse_price_tag_ocr(ocr_result, crop_path if crop_path.exists() else None)
        if LLM_REFINER_ENABLED:
            parsed = refine_with_llm(ocr_result.get("markdown_table") or ocr_result.get("plain_text", ""), parsed)
        codes = read_codes_from_image(crop_path) if crop_path.exists() else {"qr": "", "barcode": ""}
        if codes.get("barcode") and not parsed.get("barcode"):
            parsed["barcode"] = codes["barcode"]

        rows.append({
            'filename': video_path.name,
            'product_name': parsed.get('product_name', ''),
            'price_default': parsed.get('price_default', ''),
            'price_card': parsed.get('price_card', ''),
            'price_discount': parsed.get('price_discount', ''),
            'barcode': parsed.get('barcode', ''),
            'discount_amount': parsed.get('discount_amount', ''),
            'id_sku': parsed.get('id_sku', ''),
            'print_datetime': parsed.get('print_datetime', ''),
            'code': parsed.get('code', ''),
            'additional_info': (
                f"{parsed.get('additional_info', '')}; "
                f"cv_detector crop={crop_name}; "
                f"track_id={record.get('tracker_id', '')}; "
                f"sharpness={record.get('sharpness', '')}"
            ).strip('; '),
            'color': parsed.get('color', ''),
            'special_symbols': parsed.get('special_symbols', ''),
            'frame_timestamp': int(record.get('timestamp_ms', 0) or 0),
            'x_min': x,
            'y_min': y,
            'x_max': x + w,
            'y_max': y + h,
            'qr_code_barcode': codes.get('qr') or codes.get('barcode') or parsed.get('barcode', ''),
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
