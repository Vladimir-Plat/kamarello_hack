from __future__ import annotations

import json
from pathlib import Path

from app.config import LLM_MODEL_PATH, LLM_REFINER_ENABLED
from app.services.price_tag_parser import PRICE_FIELDS


def _validated(payload: dict, fallback: dict) -> dict:
    result = dict(fallback)
    for key in PRICE_FIELDS:
        value = payload.get(key)
        if isinstance(value, (str, int, float)):
            result[key] = str(value)
    return result


def refine_with_llm(ocr_markdown: str, rule_result: dict) -> dict:
    if not LLM_REFINER_ENABLED or not LLM_MODEL_PATH or not Path(LLM_MODEL_PATH).exists():
        return rule_result

    try:
        from llama_cpp import Llama

        llm = Llama(model_path=LLM_MODEL_PATH, n_ctx=2048, n_threads=2, verbose=False)
        prompt = (
            "Extract Russian retail price tag fields from OCR text only. "
            "Return one valid JSON object with exactly these keys: "
            f"{', '.join(PRICE_FIELDS.keys())}.\n"
            "Do not invent values. OCR text:\n"
            f"{ocr_markdown}\n"
            f"Rule result:\n{json.dumps(rule_result, ensure_ascii=False)}"
        )
        response = llm(prompt, max_tokens=512, temperature=0.0, stop=["</s>"])
        text = response["choices"][0]["text"]
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            return rule_result
        return _validated(json.loads(text[start : end + 1]), rule_result)
    except Exception:
        return rule_result
