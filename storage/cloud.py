"""Cloud storage abstraction.

Switch provider via CLOUD_STORAGE_PROVIDER env var.
"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

from config import get_settings


class CloudStorageBackend(ABC):
    @abstractmethod
    def upload(self, local_path: Path, remote_key: str) -> str:
        """Upload file and return the remote URI."""

    @abstractmethod
    def download(self, remote_key: str, local_path: Path) -> None:
        """Download remote_key to local_path."""

    @abstractmethod
    def signed_url(self, remote_key: str, expiry_seconds: int = 3600) -> str:
        """Return a time-limited signed URL."""

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """Return list of keys under prefix."""


class GCSBackend(CloudStorageBackend):
    def __init__(self) -> None:
        settings = get_settings()
        from google.cloud import storage as gcs
        if settings.cloud_storage_credentials:
            cred_info = json.loads(settings.cloud_storage_credentials)
            from google.oauth2 import service_account
            creds = service_account.Credentials.from_service_account_info(cred_info)
            self._client = gcs.Client(credentials=creds)
        else:
            self._client = gcs.Client()
        self._bucket_name = settings.cloud_storage_bucket
        self._bucket = self._client.bucket(self._bucket_name)

    def upload(self, local_path: Path, remote_key: str) -> str:
        blob = self._bucket.blob(remote_key)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self._bucket_name}/{remote_key}"

    def download(self, remote_key: str, local_path: Path) -> None:
        blob = self._bucket.blob(remote_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))

    def signed_url(self, remote_key: str, expiry_seconds: int = 3600) -> str:
        blob = self._bucket.blob(remote_key)
        return blob.generate_signed_url(expiration=timedelta(seconds=expiry_seconds), version="v4")

    def list_keys(self, prefix: str = "") -> List[str]:
        blobs = self._client.list_blobs(self._bucket_name, prefix=prefix)
        return [b.name for b in blobs]


class LocalFSBackend(CloudStorageBackend):
    """Local filesystem backend for development."""

    def __init__(self) -> None:
        settings = get_settings()
        self._root = Path(settings.ref_cache_path) / "local_storage"
        self._root.mkdir(parents=True, exist_ok=True)

    def upload(self, local_path: Path, remote_key: str) -> str:
        dest = self._root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(local_path, dest)
        return f"file://{dest}"

    def download(self, remote_key: str, local_path: Path) -> None:
        src = self._root / remote_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(src, local_path)

    def signed_url(self, remote_key: str, expiry_seconds: int = 3600) -> str:
        return f"file://{self._root / remote_key}"

    def list_keys(self, prefix: str = "") -> List[str]:
        base = self._root / prefix if prefix else self._root
        if not base.exists():
            return []
        return [str(p.relative_to(self._root)) for p in base.rglob("*") if p.is_file()]


_backend: Optional[CloudStorageBackend] = None


def get_storage() -> CloudStorageBackend:
    global _backend
    if _backend is None:
        provider = os.environ.get("CLOUD_STORAGE_PROVIDER", "local").lower()
        if provider == "gcs":
            _backend = GCSBackend()
        else:
            _backend = LocalFSBackend()
    return _backend
