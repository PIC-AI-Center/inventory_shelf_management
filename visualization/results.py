from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from schemas import DetectedRow

_ROW_HEIGHT = 120
_CROP_W = 180
_TILE_W = 80
_TILE_H = _ROW_HEIGHT
_FONT_SIZE = 10
_BG = (240, 240, 240)
_BORDER = (180, 180, 180)
_PLACEHOLDER_BG = (210, 210, 210)


def _try_load_image(paths: List[Path]) -> Optional[Image.Image]:
    for p in paths:
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                continue
    return None


def _row_crop(img: Image.Image, row: DetectedRow) -> Optional[Image.Image]:
    w, h = img.size
    y0 = max(0, int(row.y_min * h))
    y1 = min(h, int(row.y_max * h))
    if y1 <= y0:
        return None
    crop = img.crop((0, y0, w, y1))
    aspect = crop.width / max(crop.height, 1)
    new_w = int(_CROP_W * min(aspect, 2))
    return crop.resize((new_w, _ROW_HEIGHT))


def _product_tile(img: Optional[Image.Image], row: DetectedRow, prod_idx: int) -> Image.Image:
    prod = row.products[prod_idx]
    tile = Image.new("RGB", (_TILE_W, _TILE_H), _PLACEHOLDER_BG)
    if img is not None:
        w, h = img.size
        b = prod.bbox
        x0 = max(0, int(b.x1 * w))
        y0 = max(0, int(b.y1 * h))
        x1 = min(w, int(b.x2 * w))
        y1 = min(h, int(b.y2 * h))
        if x1 > x0 and y1 > y0:
            try:
                crop = img.crop((x0, y0, x1, y1)).resize((_TILE_W, _TILE_H))
                tile = crop
            except Exception:
                pass
    draw = ImageDraw.Draw(tile, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    label = (prod.sku_id or prod.label or "?")[:10]
    draw.rectangle([0, _TILE_H - 18, _TILE_W, _TILE_H], fill=(0, 0, 0, 160))
    draw.text((2, _TILE_H - 16), label, fill=(255, 255, 255), font=font)
    return tile


def generate_results(
    image_paths: List[Path],
    rows: List[DetectedRow],
    output_path: Path,
) -> None:
    if not rows:
        img = Image.new("RGB", (640, 100), _BG)
        ImageDraw.Draw(img).text((10, 40), "No rows detected", fill=(80, 80, 80))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="JPEG", quality=88)
        return

    source_img = _try_load_image(image_paths)

    max_products = max((len(r.products) for r in rows), default=0)
    canvas_w = _CROP_W + max_products * _TILE_W + 10
    canvas_h = len(rows) * (_ROW_HEIGHT + 4)
    canvas = Image.new("RGB", (canvas_w, canvas_h), _BG)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    for ri, row in enumerate(rows):
        y_off = ri * (_ROW_HEIGHT + 4)

        row_thumb: Optional[Image.Image] = None
        if source_img is not None:
            row_thumb = _row_crop(source_img, row)
        if row_thumb is None:
            row_thumb = Image.new("RGB", (_CROP_W, _ROW_HEIGHT), _PLACEHOLDER_BG)
            ImageDraw.Draw(row_thumb).text((4, _ROW_HEIGHT // 2 - 8), f"Row {row.row_idx}", fill=(100, 100, 100), font=font)
        canvas.paste(row_thumb, (0, y_off))

        for pi in range(len(row.products)):
            tile = _product_tile(source_img, row, pi)
            x_off = _CROP_W + pi * _TILE_W
            canvas.paste(tile, (x_off, y_off))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=88)
