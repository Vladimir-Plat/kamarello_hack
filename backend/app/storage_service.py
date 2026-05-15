from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_FILE = BASE_DIR / "storage" / "jobs.json"


def load_jobs() -> Dict[str, Dict[str, Any]]:
    if not STORAGE_FILE.exists():
        return {}

    try:
        return json.loads(STORAGE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_jobs(jobs: Dict[str, Dict[str, Any]]) -> None:
    STORAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STORAGE_FILE.write_text(
        json.dumps(jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )