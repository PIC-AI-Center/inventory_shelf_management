"""Routes: get_shelf_info."""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends

from auth import require_api_key
from planogram.loader import list_planogram_ids
from schemas import GetShelfInfoResponse, ShelfInfo
from storage.cloud import get_storage

router = APIRouter()


@router.get("/get_shelf_info", response_model=GetShelfInfoResponse, dependencies=[Depends(require_api_key)])
async def api_get_shelf_info():
    """Return all shelf/planogram codes with signed URLs for their images."""
    storage = get_storage()
    planogram_ids = list_planogram_ids()
    shelves: List[ShelfInfo] = []
    for pid in planogram_ids:
        remote_key = f"assets/{pid}.jpg"
        try:
            url = storage.signed_url(remote_key)
        except Exception:
            url = ""
        shelves.append(ShelfInfo(shelf_key=pid, signed_url=url))
    return GetShelfInfoResponse(shelves=shelves)
