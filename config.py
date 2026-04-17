from __future__ import annotations

import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_key: str = "changeme"
    fastapi_port: int = 8082
    fastapi_workers: int = 4

    cloud_storage_bucket: str = ""
    cloud_storage_credentials: str = ""

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "app_db"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    redis_url: str = "redis://localhost:6379/0"
    celery_worker_concurrency: int = 4

    classes_path: str = "classes.txt"
    save_vs_input: bool = False

    ref_cache_path: str = "cache"

    triton_http_host: str = "localhost"
    triton_http_port: int = 8000
    triton_grpc_host: str = "localhost"
    triton_grpc_port: int = 8001

    triton_shelf_model: str = "model_a"
    triton_product_model: str = "model_b"
    triton_classifier_model: str = "model_c"

    planogram_csv_dir: str = "data"

    default_top_k: int = 3


@lru_cache()
def get_settings() -> Settings:
    return Settings()
