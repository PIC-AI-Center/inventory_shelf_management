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
    (255, 128, 0),
    (128, 0, 255),
]
_FONT_SIZE = 12
_LINE_WIDTH = 2


def generate_detection_viz(
    image_path: Path,
    image_idx: int,
    rows: List[DetectedRow],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"detection_visualization_img{image_idx + 1}.jpg"

    if not image_path.exists():
        return out_path

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", _FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()

    for row in rows:
        row_color = _COLORS[row.row_idx % len(_COLORS)]
        y0 = int(row.y_min * h)
        y1 = int(row.y_max * h)
        draw.rectangle([0, y0, w - 1, y1], outline=row_color + (180,) if len(row_color) == 3 else row_color, width=1)

        for prod in row.products:
            b = prod.bbox
            x0 = int(b.x1 * w)
            y0p = int(b.y1 * h)
            x1 = int(b.x2 * w)
            y1p = int(b.y2 * h)
            draw.rectangle([x0, y0p, x1, y1p], outline=row_color, width=_LINE_WIDTH)
            label = (prod.sku_id or prod.label or "?")[:12]
            text = f"{label} {prod.score:.2f}"
            bbox = draw.textbbox((x0, y0p - _FONT_SIZE - 2), text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text((x0, y0p - _FONT_SIZE - 2), text, fill=(255, 255, 255), font=font)

    img.save(out_path, format="JPEG", quality=88)
    return out_path
