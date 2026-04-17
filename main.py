"""FastAPI application entry point."""
from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from routes.planogram import router as planogram_router
from routes.shelf import router as shelf_router
from routes.labels import router as labels_router
from routes.info import router as info_router

settings = get_settings()

app = FastAPI(
    title="Inventory Shelf Management API",
    description=(
        "Top-K planogram auto-match, shelf compliance detection, "
        "and planogram utilities powered by NVIDIA Triton + FastAPI + Celery."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(planogram_router)
app.include_router(shelf_router)
app.include_router(labels_router)
app.include_router(info_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.fastapi_port,
        workers=settings.fastapi_workers,
        log_level="info",
    )
