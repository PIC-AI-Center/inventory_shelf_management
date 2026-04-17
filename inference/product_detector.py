"""Product detection via Triton gRPC."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from config import get_settings
from inference.triton_grpc import TritonGRPCClient
from schemas import BoundingBox, DetectedProduct, DetectedRow

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    _DALI_AVAILABLE = True
except ImportError:
    _DALI_AVAILABLE = False

_INPUT_W = 640
_INPUT_H = 640
_CONF_THRESH = 0.25
_NMS_IOU = 0.45


def _load_classes(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


if _DALI_AVAILABLE:
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def _dali_product_pipeline(image_path: str):
        jpegs, _ = fn.readers.file(files=[image_path], name="Reader")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, device="gpu", resize_x=_INPUT_W, resize_y=_INPUT_H)
        images = fn.cast(images, dtype=types.FLOAT)
        images = images / 255.0
        images = fn.transpose(images, perm=[2, 0, 1])
        return images


def _preprocess(image_path: Path) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = Image.open(image_path).convert("RGB")
    orig_wh = img.size
    if _DALI_AVAILABLE:
        pipe = _dali_product_pipeline(image_path=str(image_path))
        pipe.build()
        outputs = pipe.run()
        arr = outputs[0].as_cpu().as_array()  # (1, 3, H, W)
        return arr, orig_wh
    img_resized = img.resize((_INPUT_W, _INPUT_H))
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
    return arr, orig_wh


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


class ProductDetector:
    """Detects individual products within shelf images via Triton gRPC."""

    def __init__(self) -> None:
        self._client = TritonGRPCClient()
        self._settings = get_settings()
        self._classes = _load_classes(self._settings.classes_path)

    def detect(self, image_path: Path, shelf_rows: List[DetectedRow]) -> List[DetectedRow]:
        """Run product detection and assign products to rows.

        Returns a new list of DetectedRow with .products filled in.
        """
        tensor, orig_wh = _preprocess(image_path)
        outputs = self._client.infer(
            model_name=self._settings.triton_product_model,
            inputs={"images": tensor},
            output_names=["boxes", "scores", "labels"],
        )

        raw_boxes = outputs["boxes"]
        raw_scores = outputs["scores"]
        raw_labels = outputs["labels"]

        if raw_boxes.ndim == 3:
            raw_boxes = raw_boxes[0]
            raw_scores = raw_scores[0]
            raw_labels = raw_labels[0]

        mask = raw_scores > _CONF_THRESH
        raw_boxes, raw_scores, raw_labels = raw_boxes[mask], raw_scores[mask], raw_labels[mask]

        if len(raw_boxes) > 0:
            keep = _nms(raw_boxes, raw_scores, _NMS_IOU)
            raw_boxes = raw_boxes[keep]
            raw_scores = raw_scores[keep]
            raw_labels = raw_labels[keep]

        # Assign each detection to the shelf row whose y-range overlaps most with the box center
        updated_rows = [
            DetectedRow(row_idx=r.row_idx, y_min=r.y_min, y_max=r.y_max)
            for r in shelf_rows
        ]

        for i in range(len(raw_boxes)):
            b = raw_boxes[i]
            cy = (b[1] + b[3]) / 2
            cls_idx = int(raw_labels[i])
            label = self._classes[cls_idx] if cls_idx < len(self._classes) else str(cls_idx)
            product = DetectedProduct(
                bbox=BoundingBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3])),
                score=float(raw_scores[i]),
                label=label,
            )
            # Find best matching row
            best_row = None
            best_overlap = -1.0
            for row in updated_rows:
                if row.y_min <= cy <= row.y_max:
                    overlap = min(b[3], row.y_max) - max(b[1], row.y_min)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_row = row
            if best_row is None and updated_rows:
                # fallback: assign to nearest row
                dists = [abs((r.y_min + r.y_max) / 2 - cy) for r in updated_rows]
                best_row = updated_rows[int(np.argmin(dists))]
            if best_row is not None:
                best_row.products.append(product)

        return updated_rows
