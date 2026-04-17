"""Routes: generate_label, generate_label_task/{task_id}."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from auth import require_api_key
from celery_app import celery_app
from schemas import GenerateLabelRequest, GenerateLabelResponse, LabelTaskResult
from tasks.label_tasks import generate_label

router = APIRouter()


@router.post("/generate_label", response_model=GenerateLabelResponse, dependencies=[Depends(require_api_key)])
async def api_generate_label(req: GenerateLabelRequest):
    """Enqueue a batch label translation job."""
    task = generate_label.delay(
        skus=req.skus,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
    )
    return GenerateLabelResponse(task_id=task.id)


@router.get("/generate_label_task/{task_id}", response_model=LabelTaskResult, dependencies=[Depends(require_api_key)])
async def api_generate_label_task(task_id: str):
    """Poll label translation job status."""
    task = celery_app.AsyncResult(task_id)
    resp = LabelTaskResult(task_id=task_id, status=task.state)
    if task.state == "SUCCESS":
        resp.translations = task.result
    elif task.state == "FAILURE":
        resp.error = str(task.result)
    return resp
