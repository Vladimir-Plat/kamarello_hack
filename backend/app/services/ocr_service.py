from __future__ import annotations

import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import pandas as pd

from app.config import OCR_ENABLED, OCR_LANG

logger = logging.getLogger(__name__)
_ocr_engine: Any | None = None
_ocr_lock = Lock()


def upscale(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)


def preprocess_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    norm = cv2.equalizeHist(gray)
    den = cv2.fastNlMeansDenoising(norm, None, 15, 7, 21)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(den, -1, sharp_kernel)
    adap_inv = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )

    return {
        "gray_x2": upscale(gray),
        "sharp_x2": upscale(sharp),
        "adaptive_inv_x2": upscale(adap_inv),
        "normalized_x2": upscale(norm),
        "denoised_x2": upscale(den),
    }


def _rotate(image: np.ndarray, angle: float, background: tuple[int, int, int]) -> np.ndarray:
    old_h, old_w = image.shape[:2]
    angle_rad = math.radians(angle)
    new_w = abs(math.sin(angle_rad) * old_h) + abs(math.cos(angle_rad) * old_w)
    new_h = abs(math.sin(angle_rad) * old_w) + abs(math.cos(angle_rad) * old_h)
    center = (old_w / 2, old_h / 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rot[0, 2] += (new_w - old_w) / 2
    rot[1, 2] += (new_h - old_h) / 2
    return cv2.warpAffine(image, rot, (int(round(new_w)), int(round(new_h))), borderValue=background)


def deskew_image(image: np.ndarray) -> np.ndarray:
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 20:
            return image
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if abs(angle) > 15:
            return image
        return _rotate(image, angle, (255, 255, 255))
    except Exception:
        return image


def _empty_result() -> dict[str, Any]:
    return {"raw_lines": [], "merged_rows": [], "markdown_table": "", "plain_text": ""}


def _get_ocr_engine() -> Any | None:
    global _ocr_engine
    if not OCR_ENABLED:
        return None
    if _ocr_engine is not None:
        return _ocr_engine
    with _ocr_lock:
        if _ocr_engine is not None:
            return _ocr_engine
        try:
            from paddleocr import PaddleOCR

            _ocr_engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                lang=OCR_LANG,
            )
        except TypeError:
            from paddleocr import PaddleOCR

            _ocr_engine = PaddleOCR(use_angle_cls=True, lang=OCR_LANG)
        except Exception as exc:
            logger.warning("PaddleOCR unavailable: %s", exc)
            return None
    return _ocr_engine


def _coerce_image(image_or_path: np.ndarray | str | Path) -> np.ndarray | None:
    if isinstance(image_or_path, np.ndarray):
        return image_or_path
    image = cv2.imread(str(image_or_path))
    return image


def extract_text_from_image(img: np.ndarray) -> list[dict[str, Any]]:
    engine = _get_ocr_engine()
    if engine is None:
        return []

    try:
        result = engine.predict(img) if hasattr(engine, "predict") else engine.ocr(img, cls=True)
    except Exception as exc:
        logger.warning("OCR failed on image variant: %s", exc)
        return []

    lines: list[dict[str, Any]] = []
    for res in result or []:
        if hasattr(res, "rec_texts") and hasattr(res, "rec_scores"):
            boxes = getattr(res, "rec_boxes", []) or getattr(res, "dt_polys", [])
            for txt, score, box in zip(res.rec_texts, res.rec_scores, boxes):
                lines.append({"text": str(txt).strip(), "score": float(score), "boxes": _box_to_xyxy(box)})
        elif isinstance(res, dict):
            for txt, score, box in zip(res.get("rec_texts", []), res.get("rec_scores", []), res.get("rec_boxes", [])):
                lines.append({"text": str(txt).strip(), "score": float(score), "boxes": _box_to_xyxy(box)})
        elif isinstance(res, list):
            for item in res:
                if len(item) >= 2:
                    text_score = item[1]
                    if isinstance(text_score, (list, tuple)) and len(text_score) >= 2:
                        lines.append({"text": str(text_score[0]).strip(), "score": float(text_score[1]), "boxes": _box_to_xyxy(item[0])})

    return [line for line in lines if line["text"]]


def _box_to_xyxy(box: Any) -> list[int]:
    arr = np.array(box).astype(float).reshape(-1)
    if arr.size >= 8:
        xs = arr[0::2]
        ys = arr[1::2]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    if arr.size >= 4:
        return [int(v) for v in arr[:4]]
    return [0, 0, 0, 0]


def run_multi_variant_ocr(image: np.ndarray | str | Path) -> dict[str, list[dict[str, Any]]]:
    source = _coerce_image(image)
    if source is None:
        return {}
    source = deskew_image(source)
    results = {}
    for name, variant in preprocess_variants(source).items():
        if len(variant.shape) == 2:
            variant = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
        results[name] = extract_text_from_image(variant)
    return results


def merge_text_by_rows(
    results: dict[str, list[dict[str, Any]]],
    y_threshold: int = 20,
    score_threshold: float = 0.5,
    global_y_merge_threshold: int = 10,
) -> pd.DataFrame:
    prep_row_data: dict[str, dict[float, str]] = {}
    for prep_type, boxes in results.items():
        valid = [b for b in boxes if float(b.get("score", 0)) > score_threshold]
        valid.sort(key=lambda b: (b.get("boxes", [0, 0, 0, 0])[1], b.get("boxes", [0, 0, 0, 0])[0]))
        rows: dict[float, str] = {}
        current_y = None
        current_text: list[str] = []
        for box in valid:
            coords = box.get("boxes", [0, 0, 0, 0])
            center_y = (coords[1] + coords[3]) / 2
            if current_y is None or abs(center_y - current_y) > y_threshold:
                if current_y is not None:
                    rows[round(current_y, 1)] = " ".join(current_text)
                current_y = center_y
                current_text = [str(box["text"])]
            else:
                current_text.append(str(box["text"]))
        if current_y is not None:
            rows[round(current_y, 1)] = " ".join(current_text)
        prep_row_data[prep_type] = rows

    y_positions = sorted({y for rows in prep_row_data.values() for y in rows})
    groups: list[tuple[float, list[float]]] = []
    for y in y_positions:
        if not groups or abs(y - groups[-1][0]) > global_y_merge_threshold:
            groups.append((y, [y]))
        else:
            groups[-1][1].append(y)
            groups[-1] = (sum(groups[-1][1]) / len(groups[-1][1]), groups[-1][1])

    combined = []
    for merged_y, ys in groups:
        row = {"y_position": round(merged_y, 1)}
        for prep_type, rows in prep_row_data.items():
            text = " ".join(rows[y] for y in ys if y in rows)
            row[prep_type] = text or None
        combined.append(row)

    return pd.DataFrame(combined).set_index("y_position") if combined else pd.DataFrame()


def ocr_result_to_payload(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    raw_lines = []
    seen = set()
    for variant, lines in results.items():
        for line in lines:
            key = (line["text"], tuple(line.get("boxes", [])))
            if key not in seen:
                seen.add(key)
                raw_lines.append({**line, "variant": variant})

    rows_df = merge_text_by_rows(results)
    merged_rows = []
    if not rows_df.empty:
        for _, row in rows_df.iterrows():
            values = [str(v) for v in row.tolist() if pd.notna(v) and str(v).strip()]
            if values:
                merged_rows.append(" | ".join(dict.fromkeys(values)))

    plain_lines = merged_rows or [line["text"] for line in raw_lines]
    markdown = rows_df.to_markdown() if not rows_df.empty else ""
    return {
        "raw_lines": raw_lines,
        "merged_rows": merged_rows,
        "markdown_table": markdown,
        "plain_text": "\n".join(plain_lines),
    }


def extract_ocr_payload(image: np.ndarray | str | Path) -> dict[str, Any]:
    if not OCR_ENABLED:
        return _empty_result()
    try:
        return ocr_result_to_payload(run_multi_variant_ocr(image))
    except Exception as exc:
        logger.warning("OCR pipeline failed: %s", exc)
        return _empty_result()
