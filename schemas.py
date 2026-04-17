from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectedProduct(BaseModel):
    bbox: BoundingBox
    score: float
    label: str
    sku_id: Optional[str] = None
    embedding: Optional[List[float]] = None


class DetectedRow(BaseModel):
    row_idx: int
    y_min: float
    y_max: float
    products: List[DetectedProduct] = []


class MatchPlanogramRequest(BaseModel):
    images: List[str]
    image_orientations: Optional[List[int]] = None
    top_k: int = 3
    store_id: Optional[str] = None


class RowMatch(BaseModel):
    detected_row_idx: int
    planogram_row_idx: int
    similarity: float


class PlanogramCandidate(BaseModel):
    planogram_id: str
    score: float
    matched_rows: List[RowMatch] = []


class MatchMeta(BaseModel):
    num_images: int
    num_detected_rows: int
    total_labels: int


class MatchPlanogramResponse(BaseModel):
    match_id: str
    top_k: int
    matches: List[PlanogramCandidate]
    meta: MatchMeta


class ConfirmPlanogramRequest(BaseModel):
    match_id: str
    planogram_id: str
    lang: str = "en"
    create_visualizations: bool = True
    delete_after: bool = False


class ConfirmMeta(BaseModel):
    rows_in_planogram: int
    assets: List[str] = []


class ConfirmPlanogramResponse(BaseModel):
    match_id: str
    planogram_id: str
    result: Dict[str, Any]
    meta: ConfirmMeta


class CheckShelfRequest(BaseModel):
    images: List[str]
    image_orientations: Optional[List[int]] = None
    shelf_key: str
    lang: str = "en"


class ShelfRowCompliance(BaseModel):
    row_idx: int
    planogram_row_idx: Optional[int] = None
    expected_skus: List[str] = []
    detected_skus: List[str] = []
    compliance_score: float = 0.0
    missing: List[str] = []
    unexpected: List[str] = []


class CheckShelfResult(BaseModel):
    shelf_key: str
    overall_compliance: float
    rows: List[ShelfRowCompliance] = []
    assets: List[str] = []
    lang: str = "en"


class AsyncTaskResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class CreatePlanogram2DRequest(BaseModel):
    planogram_id: str
    rows: List[List[str]]
    cell_width: int = 80
    cell_height: int = 60
    font_size: int = 10


class CreatePlanogram2DResponse(BaseModel):
    planogram_id: str
    image_url: str


class ShelfInfo(BaseModel):
    shelf_key: str
    signed_url: str
    metadata: Dict[str, Any] = {}


class GetShelfInfoResponse(BaseModel):
    shelves: List[ShelfInfo]


class GenerateLabelRequest(BaseModel):
    skus: List[str]
    source_lang: str = "en"
    target_lang: str = "en"


class GenerateLabelResponse(BaseModel):
    task_id: str


class LabelTaskResult(BaseModel):
    task_id: str
    status: str
    translations: Optional[Dict[str, str]] = None
    error: Optional[str] = None
