from __future__ import annotations

from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from schemas import DetectedRow

_COLORS = [
    (255, 80, 80),
    (80, 200, 80),
    (80, 80, 255),
    (255, 200, 0),
    (200, 0, 255),
    (0, 200, 200),
]
_THUMB_W = 640
_FONT_SIZE = 14


def generate_simplified(
    image_paths: List[Path],
    rows: List[DetectedRow],
    output_path: Path,
) -> None:
    src_path = next((p for p in image_paths if p.exists()), None)
    if src_path is None:
        _render_placeholder(rows, output_path)
        return

    img = Image.open(src_path).convert("RGB")
    w, h = img.size
    scale = _THUMB_W / w
    thumb = img.resize((_THUMB_W, int(h * scale)))
    tw, th = thumb.size

    draw = ImageDraw.Draw(thumb, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    for row in rows:
        color = _COLORS[row.row_idx % len(_COLORS)]
        y0 = int(row.y_min * th)
        y1 = int(row.y_max * th)
        draw.rectangle([0, y0, tw - 1, y1], outline=color + (220,), width=2)
        draw.text((4, y0 + 2), f"Row {row.row_idx}", fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    thumb.save(output_path, format="JPEG", quality=88)


def _render_placeholder(rows: List[DetectedRow], output_path: Path) -> None:
    img = Image.new("RGB", (_THUMB_W, 480), (230, 230, 230))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"No source image — {len(rows)} rows detected", fill=(80, 80, 80))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="JPEG", quality=88)
