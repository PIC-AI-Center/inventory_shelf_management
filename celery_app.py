from celery import Celery
from config import get_settings

settings = get_settings()

celery_app = Celery(
    "app",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "tasks.planogram_tasks",
        "tasks.shelf_tasks",
        "tasks.label_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=settings.celery_worker_concurrency,
    task_track_started=True,
    result_expires=3600,
)
