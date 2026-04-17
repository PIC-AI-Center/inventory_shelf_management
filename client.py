from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class ShelfAPIClient:
    def __init__(self, base_url: str = "http://localhost:8082", api_key: str = "", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"x-api-key": api_key})

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        data = Path(path).read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def match_planogram(
        self,
        image_paths: List[str | Path],
        top_k: int = 3,
        store_id: str = "",
        image_orientations: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        images = [self._encode_image(p) for p in image_paths]
        payload: Dict[str, Any] = {"images": images, "top_k": top_k}
        if store_id:
            payload["store_id"] = store_id
        if image_orientations:
            payload["image_orientations"] = image_orientations
        resp = self._session.post(self._url("/match_planogram"), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def choose_by_highest_score(self, match_response: Dict[str, Any]) -> str:
        matches = match_response.get("matches", [])
        if not matches:
            raise ValueError("No matches in response")
        return max(matches, key=lambda m: m["score"])["planogram_id"]

    def confirm_planogram(
        self,
        match_id: str,
        planogram_id: str,
        lang: str = "en",
        create_visualizations: bool = True,
        delete_after: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "match_id": match_id,
            "planogram_id": planogram_id,
            "lang": lang,
            "create_visualizations": create_visualizations,
            "delete_after": delete_after,
        }
        resp = self._session.post(self._url("/confirm_planogram"), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def create_planogram_2d(
        self,
        planogram_id: str,
        rows: List[List[str]],
        cell_width: int = 80,
        cell_height: int = 60,
        font_size: int = 10,
    ) -> Dict[str, Any]:
        payload = {
            "planogram_id": planogram_id,
            "rows": rows,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "font_size": font_size,
        }
        resp = self._session.post(self._url("/create_planogram_2d"), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def check_shelf(
        self,
        image_paths: List[str | Path],
        shelf_key: str,
        lang: str = "en",
        image_orientations: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        images = [self._encode_image(p) for p in image_paths]
        payload: Dict[str, Any] = {"images": images, "shelf_key": shelf_key, "lang": lang}
        if image_orientations:
            payload["image_orientations"] = image_orientations
        resp = self._session.post(self._url("/check_shelf"), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def check_shelf_async(
        self,
        image_paths: List[str | Path],
        shelf_key: str,
        lang: str = "en",
        image_orientations: Optional[List[int]] = None,
    ) -> str:
        images = [self._encode_image(p) for p in image_paths]
        payload: Dict[str, Any] = {"images": images, "shelf_key": shelf_key, "lang": lang}
        if image_orientations:
            payload["image_orientations"] = image_orientations
        resp = self._session.post(self._url("/check_shelf_async"), json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["task_id"]

    def get_task(self, task_id: str) -> Dict[str, Any]:
        resp = self._session.get(self._url(f"/task/{task_id}"), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def wait_for_task(self, task_id: str, poll_interval: float = 2.0, max_wait: float = 300.0) -> Dict[str, Any]:
        start = time.monotonic()
        while True:
            result = self.get_task(task_id)
            if result["status"] in ("SUCCESS", "FAILURE"):
                return result
            if time.monotonic() - start > max_wait:
                raise TimeoutError(f"Task {task_id} did not complete within {max_wait}s")
            time.sleep(poll_interval)

    def get_shelf_info(self) -> Dict[str, Any]:
        resp = self._session.get(self._url("/get_shelf_info"), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def generate_label(
        self,
        skus: List[str],
        source_lang: str = "en",
        target_lang: str = "en",
    ) -> str:
        payload = {"skus": skus, "source_lang": source_lang, "target_lang": target_lang}
        resp = self._session.post(self._url("/generate_label"), json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["task_id"]

    def get_label_task(self, task_id: str) -> Dict[str, Any]:
        resp = self._session.get(self._url(f"/generate_label_task/{task_id}"), timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        try:
            resp = self._session.get(self._url("/health"), timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
