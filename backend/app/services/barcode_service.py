from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _coerce_image(image: np.ndarray | str | Path) -> np.ndarray | None:
    if isinstance(image, np.ndarray):
        return image
    return cv2.imread(str(image))


def read_codes_from_image(image: np.ndarray | str | Path) -> dict[str, str]:
    source = _coerce_image(image)
    if source is None:
        return {"qr": "", "barcode": ""}

    qr_value = ""
    try:
        detector = cv2.QRCodeDetector()
        value, _, _ = detector.detectAndDecode(source)
        qr_value = value or ""
    except Exception:
        qr_value = ""

    return {"qr": qr_value, "barcode": ""}
