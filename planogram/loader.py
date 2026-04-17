"""Planogram data loader."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from config import get_settings


def _csv_dir() -> Path:
    return Path(get_settings().planogram_csv_dir)


def list_planogram_ids() -> List[str]:
    d = _csv_dir()
    if not d.exists():
        return []
    return [p.stem for p in sorted(d.glob("*.csv"))]


def load_planogram(planogram_id: str) -> List[List[str]]:
    """Return list of rows; each row is a list of SKU IDs/names."""
    p = _csv_dir() / f"{planogram_id}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Planogram not found: {planogram_id}")
    rows: List[List[str]] = []
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append([cell.strip() for cell in row if cell.strip()])
    return rows


def all_planograms() -> Dict[str, List[List[str]]]:
    return {pid: load_planogram(pid) for pid in list_planogram_ids()}
