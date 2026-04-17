"""Shelf row detection via Triton HTTP.

Preprocessing uses NVIDIA DALI when available, otherwise falls back to PIL.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import get_settings
from inference.triton_http import TritonHTTPClient
from schemas import BoundingBox, DetectedRow

# DALI optional — GPU-accelerated JPEG decode + resize + normalise
try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
    _DALI_AVAILABLE = True
except ImportError:
    _DALI_AVAILABLE = False

from PIL import Image


_INPUT_W = 640
_INPUT_H = 640
_CONF_THRESH = 0.25
_NMS_IOU = 0.45


# ---------------------------------------------------------------------------
# DALI pipeline (used when GPU is available)
# ---------------------------------------------------------------------------

if _DALI_AVAILABLE:
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def _dali_preprocess_pipeline(image_path: str):
        # Read raw JPEG bytes from disk
        jpegs, _ = fn.readers.file(files=[image_path], name="Reader")
        # GPU-accelerated JPEG decode directly into RGB
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        # Resize to model input size on GPU
        images = fn.resize(images, device="gpu", resize_x=_INPUT_W, resize_y=_INPUT_H)
        # Cast to float, scale 0-1, layout HWC -> CHW
        images = fn.cast(images, dtype=types.FLOAT)
        images = images / 255.0
        images = fn.transpose(images, perm=[2, 0, 1])  # HWC -> CHW
        return images


def _preprocess_dali(image_path: Path) -> np.ndarray:
    """GPU decode + resize via DALI. Returns (1, 3, H, W) FP32 on CPU."""
    pipe = _dali_preprocess_pipeline(image_path=str(image_path))
    pipe.build()
    outputs = pipe.run()
    # outputs[0] is a TensorListGPU — copy to CPU numpy
    tensor = outputs[0].as_cpu().as_array()  # (1, 3, H, W)
    return tensor


def _preprocess_pil(image_path: Path) -> np.ndarray:
    """CPU fallback: PIL decode + resize. Returns (1, 3, H, W) FP32."""
    img = Image.open(image_path).convert("RGB").resize((_INPUT_W, _INPUT_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    return arr[np.newaxis, ...]


def _preprocess(image_path: Path) -> np.ndarray:
    if _DALI_AVAILABLE:
        return _preprocess_dali(image_path)
    return _preprocess_pil(image_path)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Simple CPU NMS."""
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
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


class ShelfDetector:
    """Detects shelf rows in an image using YOLO via Triton HTTP."""

    def __init__(self) -> None:
        self._client = TritonHTTPClient()
        self._settings = get_settings()

    def detect(self, image_path: Path, original_wh: Tuple[int, int] | None = None) -> List[DetectedRow]:
        """Run shelf detection and return rows sorted top-to-bottom."""
        tensor = _preprocess(image_path)
        outputs = self._client.infer_numpy(
            model_name=self._settings.triton_shelf_model,
            input_name="images",
            array=tensor,
            output_names=["output0"],
        )
        raw = outputs["output0"]
        preds = raw[0]
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)

        conf = preds[:, 4]
        mask = conf > _CONF_THRESH
        preds = preds[mask]
        if len(preds) == 0:
            return []

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        scores = preds[:, 4]

        keep = _nms(boxes, scores, _NMS_IOU)
        boxes = boxes[keep]
        scores = scores[keep]

        order = np.argsort(boxes[:, 1])
        rows: List[DetectedRow] = []
        for idx, i in enumerate(order):
            rows.append(
                DetectedRow(
                    row_idx=idx,
                    y_min=float(boxes[i, 1]),
                    y_max=float(boxes[i, 3]),
                )
            )
        return rows
