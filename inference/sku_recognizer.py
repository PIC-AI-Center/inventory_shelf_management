"""SKU recognition via Triton gRPC.

Compares crop embeddings to a library of reference embeddings to identify SKUs.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config import get_settings
from inference.triton_grpc import TritonGRPCClient
from schemas import DetectedProduct, DetectedRow

try:
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali import pipeline_def
    _DALI_AVAILABLE = True
except ImportError:
    _DALI_AVAILABLE = False

_CROP_W = 224
_CROP_H = 224

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

if _DALI_AVAILABLE:
    @pipeline_def(batch_size=1, num_threads=2, device_id=0)
    def _dali_classifier_pipeline(image_path: str):
        jpegs, _ = fn.readers.file(files=[image_path], name="Reader")
        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(images, device="gpu", resize_x=_CROP_W, resize_y=_CROP_H)
        images = fn.cast(images, dtype=types.FLOAT)
        images = images / 255.0
        # ImageNet normalisation on GPU
        mean = fn.constant(fdata=[0.485, 0.456, 0.406], shape=[1, 1, 3], device="gpu")
        std = fn.constant(fdata=[0.229, 0.224, 0.225], shape=[1, 1, 3], device="gpu")
        images = (images - mean) / std
        images = fn.transpose(images, perm=[2, 0, 1])  # HWC -> CHW
        return images


def _preprocess_crop(crop: Image.Image) -> np.ndarray:
    """Preprocess a PIL crop to (1, 3, 224, 224) FP32 with ImageNet normalisation."""
    if _DALI_AVAILABLE:
        # Save crop to a temp file so DALI can read it
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            crop.convert("RGB").save(tmp.name, format="JPEG")
            tmp_path = tmp.name
        try:
            pipe = _dali_classifier_pipeline(image_path=tmp_path)
            pipe.build()
            outputs = pipe.run()
            return outputs[0].as_cpu().as_array()  # (1, 3, 224, 224)
        finally:
            os.unlink(tmp_path)
    # CPU fallback
    img = crop.convert("RGB").resize((_CROP_W, _CROP_H))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - _MEAN) / _STD
    return arr[np.newaxis, ...]


class SKURecognizer:
    """Uses DINO embeddings to match crops to known SKU IDs."""

    def __init__(self, library_path: str | None = None) -> None:
        self._client = TritonGRPCClient()
        self._settings = get_settings()
        self._library: Dict[str, np.ndarray] = {}  # sku_id -> embedding
        if library_path:
            self.load_library(library_path)

    def load_library(self, path: str) -> None:
        """Load a pickled dict of {sku_id: embedding_vector}."""
        p = Path(path)
        if p.exists():
            with open(p, "rb") as f:
                self._library = pickle.load(f)

    def embed(self, crop: Image.Image) -> np.ndarray:
        """Return the DINO embedding for a single crop."""
        tensor = _preprocess_crop(crop)
        outputs = self._client.infer(
            model_name=self._settings.triton_classifier_model,
            inputs={"input": tensor},
            output_names=["output"],
        )
        emb = outputs["output"][0]
        norm = np.linalg.norm(emb) + 1e-8
        return emb / norm

    def match(self, embedding: np.ndarray, top_k: int = 1) -> List[Tuple[str, float]]:
        """Return top-k (sku_id, cosine_similarity) pairs from the library."""
        if not self._library:
            return []
        ids = list(self._library.keys())
        mat = np.stack(list(self._library.values()))  # (N, D)
        sims = mat @ embedding  # cosine similarity (embeddings are L2-normed)
        order = np.argsort(sims)[::-1][:top_k]
        return [(ids[i], float(sims[i])) for i in order]

    def recognise_rows(
        self,
        image_path: Path,
        rows: List[DetectedRow],
        top_k: int = 1,
    ) -> List[DetectedRow]:
        """Fill in sku_id and embedding for each product in the rows."""
        if not self._library and not self._settings.triton_classifier_model:
            return rows

        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        updated: List[DetectedRow] = []
        for row in rows:
            new_products = []
            for prod in row.products:
                b = prod.bbox
                x1 = int(b.x1 * w)
                y1 = int(b.y1 * h)
                x2 = int(b.x2 * w)
                y2 = int(b.y2 * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    new_products.append(prod)
                    continue
                crop = img.crop((x1, y1, x2, y2))
                emb = self.embed(crop)
                matches = self.match(emb, top_k=top_k)
                sku_id = matches[0][0] if matches else None
                new_prod = prod.model_copy(
                    update={"sku_id": sku_id, "embedding": emb.tolist()}
                )
                new_products.append(new_prod)
            updated.append(
                DetectedRow(
                    row_idx=row.row_idx,
                    y_min=row.y_min,
                    y_max=row.y_max,
                    products=new_products,
                )
            )
        return updated
