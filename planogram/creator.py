"""Create a 2D planogram image from a row/column SKU grid."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

from config import get_settings


_BG_COLOR = (245, 245, 245)
_CELL_BG = (255, 255, 255)
_BORDER_COLOR = (180, 180, 180)
_TEXT_COLOR = (30, 30, 30)
_HEADER_BG = (60, 100, 180)
_HEADER_TEXT = (255, 255, 255)


def create_planogram_image(
    rows: List[List[str]],
    cell_width: int = 80,
    cell_height: int = 60,
    font_size: int = 10,
) -> bytes:
    """Render a grid of SKU cells and return JPEG bytes."""
    if not rows:
        raise ValueError("rows must not be empty")

    max_cols = max(len(r) for r in rows)
    n_rows = len(rows)
    margin = 4

    img_w = max_cols * cell_width + 2 * margin
    img_h = n_rows * cell_height + 2 * margin
    img = Image.new("RGB", (img_w, img_h), color=_BG_COLOR)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for ri, row in enumerate(rows):
        for ci, sku in enumerate(row):
            x0 = margin + ci * cell_width
            y0 = margin + ri * cell_height
            x1 = x0 + cell_width - 1
            y1 = y0 + cell_height - 1
            draw.rectangle([x0, y0, x1, y1], fill=_CELL_BG, outline=_BORDER_COLOR)
            # Wrap text
            text = sku[:16] + "…" if len(sku) > 16 else sku
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x0 + (cell_width - tw) // 2
            ty = y0 + (cell_height - th) // 2
            draw.text((tx, ty), text, fill=_TEXT_COLOR, font=font)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def save_planogram_image(
    planogram_id: str,
    rows: List[List[str]],
    cell_width: int = 80,
    cell_height: int = 60,
    font_size: int = 10,
) -> Path:
    """Save planogram image to the planogram CSV dir and return the path."""
    settings = get_settings()
    out_dir = Path(settings.planogram_csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{planogram_id}.jpg"
    jpeg_bytes = create_planogram_image(rows, cell_width, cell_height, font_size)
    out_path.write_bytes(jpeg_bytes)
    return out_path
