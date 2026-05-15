from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    filename: str
    rows_count: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    preview_rows: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    csv_url: Optional[str] = None
    xlsx_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobHistoryResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    filename: str
    rows_count: int = 0
    error: Optional[str] = None
    csv_url: Optional[str] = None
    xlsx_url: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None