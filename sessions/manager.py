from __future__ import annotations

import base64
import json
import os
import pickle
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import get_settings


def _sessions_root() -> Path:
    settings = get_settings()
    root = Path(settings.ref_cache_path) / "match_sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def new_session_id() -> str:
    return str(uuid.uuid4())


def session_dir(match_id: str) -> Path:
    return _sessions_root() / match_id


def viz_dir(match_id: str) -> Path:
    d = session_dir(match_id) / "visualizations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_images(match_id: str, b64_images: List[str]) -> List[Path]:
    sdir = session_dir(match_id)
    sdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for idx, data_uri in enumerate(b64_images):
        if "," in data_uri:
            _, encoded = data_uri.split(",", 1)
        else:
            encoded = data_uri
        raw = base64.b64decode(encoded)
        p = sdir / f"img_{idx:02d}.jpg"
        p.write_bytes(raw)
        paths.append(p)
    return paths


def load_image_paths(match_id: str) -> List[Path]:
    sdir = session_dir(match_id)
    return sorted(sdir.glob("img_*.jpg"))


def save_detected(match_id: str, data: Any) -> None:
    p = session_dir(match_id) / "detected.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)


def load_detected(match_id: str) -> Any:
    p = session_dir(match_id) / "detected.pkl"
    with open(p, "rb") as f:
        return pickle.load(f)


def save_meta(match_id: str, meta: Dict[str, Any]) -> None:
    p = session_dir(match_id) / "meta.json"
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)


def load_meta(match_id: str) -> Dict[str, Any]:
    p = session_dir(match_id) / "meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def list_assets(match_id: str) -> List[str]:
    vdir = session_dir(match_id) / "visualizations"
    if not vdir.exists():
        return []
    return [p.name for p in sorted(vdir.iterdir()) if p.is_file()]


def delete_session(match_id: str) -> None:
    d = session_dir(match_id)
    if d.exists():
        shutil.rmtree(d)
