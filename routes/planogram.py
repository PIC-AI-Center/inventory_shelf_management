"""Routes: match_planogram, confirm_planogram, create_planogram_2d."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from auth import require_api_key
from celery_app import celery_app
from config import get_settings
from schemas import (
    ConfirmMeta,
    ConfirmPlanogramRequest,
    ConfirmPlanogramResponse,
    CreatePlanogram2DRequest,
    CreatePlanogram2DResponse,
    MatchMeta,
    MatchPlanogramRequest,
    MatchPlanogramResponse,
    PlanogramCandidate,
    RowMatch,
)
from sessions import manager as sess
from storage.cloud import get_storage
from tasks.planogram_tasks import finalize_planogram_match, match_planogram

router = APIRouter()


@router.post("/match_planogram", response_model=MatchPlanogramResponse)
async def api_match_planogram(req: MatchPlanogramRequest):
    """Upload shelf images and receive Top-K planogram candidates."""
    settings = get_settings()
    match_id = sess.new_session_id()
    orientations = req.image_orientations or [0] * len(req.images)

    task = match_planogram.delay(
        match_id=match_id,
        b64_images=req.images,
        image_orientations=orientations,
        top_k=req.top_k,
        store_id=req.store_id or "",
    )
    result = task.get(timeout=300)

    return MatchPlanogramResponse(
        match_id=result["match_id"],
        top_k=result["top_k"],
        matches=[PlanogramCandidate(**c) for c in result["matches"]],
        meta=MatchMeta(**result["meta"]),
    )


@router.post("/confirm_planogram", response_model=ConfirmPlanogramResponse)
async def api_confirm_planogram(req: ConfirmPlanogramRequest):
    """Confirm a planogram candidate; generate final results and visualizations."""
    task = finalize_planogram_match.delay(
        match_id=req.match_id,
        planogram_id=req.planogram_id,
        lang=req.lang,
        create_visualizations=req.create_visualizations,
        delete_after=req.delete_after,
    )
    result = task.get(timeout=300)

    return ConfirmPlanogramResponse(
        match_id=result["match_id"],
        planogram_id=result["planogram_id"],
        result=result["result"],
        meta=ConfirmMeta(**result["meta"]),
    )


@router.post("/create_planogram_2d", response_model=CreatePlanogram2DResponse, dependencies=[Depends(require_api_key)])
async def api_create_planogram_2d(req: CreatePlanogram2DRequest):
    """Create a 2D planogram image and upload it to cloud storage."""
    from planogram.creator import create_planogram_image
    storage = get_storage()
    jpeg_bytes = create_planogram_image(
        rows=req.rows,
        cell_width=req.cell_width,
        cell_height=req.cell_height,
        font_size=req.font_size,
    )
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(jpeg_bytes)
        tmp_path = tmp.name
    try:
        remote_key = f"assets/{req.planogram_id}.jpg"
        url = storage.signed_url(remote_key)
        from pathlib import Path
        storage.upload(Path(tmp_path), remote_key)
        url = storage.signed_url(remote_key)
    finally:
        os.unlink(tmp_path)

    return CreatePlanogram2DResponse(planogram_id=req.planogram_id, image_url=url)
