"""Routes: check_shelf, check_shelf_async, task/{task_id}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from celery_app import celery_app
from schemas import (
    AsyncTaskResponse,
    CheckShelfRequest,
    CheckShelfResult,
    ShelfRowCompliance,
    TaskStatusResponse,
)
from tasks.shelf_tasks import process_check_shelf

router = APIRouter()


@router.post("/check_shelf", response_model=CheckShelfResult)
async def api_check_shelf(req: CheckShelfRequest):
    """Synchronous shelf compliance check (blocks until complete)."""
    orientations = req.image_orientations or [0] * len(req.images)
    task = process_check_shelf.delay(
        b64_images=req.images,
        image_orientations=orientations,
        shelf_key=req.shelf_key,
        lang=req.lang,
    )
    result = task.get(timeout=300)
    return CheckShelfResult(
        shelf_key=result["shelf_key"],
        overall_compliance=result["overall_compliance"],
        rows=[ShelfRowCompliance(**r) for r in result["rows"]],
        assets=result.get("assets", []),
        lang=result.get("lang", "en"),
    )


@router.post("/check_shelf_async", response_model=AsyncTaskResponse)
async def api_check_shelf_async(req: CheckShelfRequest):
    """Enqueue shelf compliance check; returns task_id for polling."""
    orientations = req.image_orientations or [0] * len(req.images)
    task = process_check_shelf.delay(
        b64_images=req.images,
        image_orientations=orientations,
        shelf_key=req.shelf_key,
        lang=req.lang,
    )
    return AsyncTaskResponse(task_id=task.id)


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def api_task_status(task_id: str):
    """Poll Celery task status/result."""
    task = celery_app.AsyncResult(task_id)
    response = TaskStatusResponse(task_id=task_id, status=task.state)
    if task.state == "SUCCESS":
        response.result = task.result
    elif task.state == "FAILURE":
        response.error = str(task.result)
    elif task.state in ("STARTED", "RETRY"):
        response.result = task.info  # progress meta
    return response
