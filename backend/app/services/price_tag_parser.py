from __future__ import annotations

import re
from pathlib import Path
from typing import Any

PRICE_FIELDS = {
    "product_name": "",
    "price_default": "",
    "price_card": "",
    "price_discount": "",
    "barcode": "",
    "discount_amount": "",
    "id_sku": "",
    "print_datetime": "",
    "code": "",
    "additional_info": "",
    "color": "",
    "special_symbols": "",
}

PRICE_RE = re.compile(r"(?<!\d)(\d{1,4})\s*[,.]?\s*(\d{2})?(?!\d)")
BARCODE_RE = re.compile(r"(?<!\d)(\d{8,14})(?!\d)")
DISCOUNT_RE = re.compile(r"-\s?\d{1,2}\s?%")
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}(?:\s+\d{1,2}:\d{2})?\b")


def _lines(ocr_result: dict[str, Any]) -> list[str]:
    merged = [str(x).strip() for x in ocr_result.get("merged_rows", []) if str(x).strip()]
    if merged:
        return merged
    raw = []
    for item in ocr_result.get("raw_lines", []):
        text = str(item.get("text", "")).strip()
        if text:
            raw.append(text)
    plain = [x.strip() for x in str(ocr_result.get("plain_text", "")).splitlines() if x.strip()]
    return raw or plain


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", " ")).strip()


def _price_candidates(line: str) -> list[str]:
    prices = []
    for rub, kop in PRICE_RE.findall(line.replace("O", "0").replace("о", "0")):
        value = f"{rub}.{kop}" if kop else rub
        try:
            number = float(value)
        except ValueError:
            continue
        if 1 <= number <= 99999 and not (kop is None and len(rub) >= 8):
            prices.append(value)
    return prices


def _first_price(line: str) -> str:
    prices = _price_candidates(line)
    return prices[0] if prices else ""


def _largest_price(lines: list[str]) -> str:
    scored = []
    for index, line in enumerate(lines):
        lower_weight = index / max(len(lines), 1)
        for price in _price_candidates(line):
            try:
                scored.append((lower_weight, float(price), price))
            except ValueError:
                pass
    if not scored:
        return ""
    return sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)[0][2]


def parse_price_tag_ocr(ocr_result: dict[str, Any], crop_path: Path | None = None) -> dict[str, str]:
    result = dict(PRICE_FIELDS)
    lines = [_norm(line) for line in _lines(ocr_result)]
    text = "\n".join(lines)
    text_lower = text.lower()

    discount = DISCOUNT_RE.search(text)
    if discount:
        result["discount_amount"] = discount.group(0).replace(" ", "")

    barcode = BARCODE_RE.search(re.sub(r"\D(?=\d{8,14}\D)", " ", text))
    if barcode:
        result["barcode"] = barcode.group(1)

    dates = DATE_RE.findall(text)
    if dates:
        result["print_datetime"] = dates[-1]

    for line in lines:
        lower = line.lower()
        price = _first_price(line)
        if not price:
            continue
        if any(marker in lower for marker in ("без карты", "базовая", "цена без карты", "без карт")):
            result["price_default"] = price
        elif any(marker in lower for marker in ("с карт", "по карт", "карта")):
            result["price_card"] = price

    result["price_discount"] = _largest_price(lines)
    if not result["price_card"] and "с карт" in text_lower:
        result["price_card"] = result["price_discount"]

    info_markers = (
        "0.75", "0,75", "0.25", "0,25", "0.5", "0,5", "л", "ml", "мл",
        "франц", "герман", "росси", "сух", "бел", "красн", "безалкоголь",
    )
    info = [line for line in lines if any(marker in line.lower() for marker in info_markers)]
    result["additional_info"] = "; ".join(dict.fromkeys(info[:4]))

    for color in ("красный", "красн", "белый", "бел", "розовый", "роз", "бордо", "игристое"):
        if color in text_lower:
            result["color"] = {
                "красн": "красный",
                "бел": "белый",
                "роз": "розовый",
            }.get(color, color)
            break

    special = re.search(r"\b\d+\s*по\s*цен[еы]\s*\d+\b", text_lower)
    if special:
        result["special_symbols"] = special.group(0)

    top_lines = []
    stop_markers = ("%", "карт", "цена", "руб", "barcode", "штрих", "qr")
    for line in lines[:8]:
        lower = line.lower()
        if BARCODE_RE.search(line) or DISCOUNT_RE.search(line) or _price_candidates(line):
            continue
        if any(marker in lower for marker in stop_markers):
            continue
        top_lines.append(line)
    result["product_name"] = " ".join(top_lines[:4])

    code_source_lines = [
        line for line in lines[:8]
        if not _price_candidates(line) and not DISCOUNT_RE.search(line) and not BARCODE_RE.search(line)
    ]
    short_codes = re.findall(r"(?<!\d)(\d{3,7})(?!\d)", "\n".join(code_source_lines))
    if short_codes:
        result["id_sku"] = short_codes[0]
        result["code"] = short_codes[-1]

    return result
