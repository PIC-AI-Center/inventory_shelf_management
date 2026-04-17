from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

from celery_app import celery_app
from inference.shelf_detector import ShelfDetector
from inference.product_detector import ProductDetector
from inference.sku_recognizer import SKURecognizer
from planogram.compliance import check_compliance
from sessions import manager as sess
from visualization.simplified import generate_simplified
from visualization.results import generate_results
from schemas import DetectedRow


@celery_app.task(name="tasks.shelf_tasks.process_check_shelf", bind=True)
def process_check_shelf(
    self,
    b64_images: List[str],
    image_orientations: List[int],
    shelf_key: str,
    lang: str,
) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    self.update_state(state="STARTED", meta={"step": "saving_images"})

    image_paths = sess.save_images(session_id, b64_images)

    shelf_det = ShelfDetector()
    prod_det = ProductDetector()
    sku_rec = SKURecognizer()

    all_rows: List[DetectedRow] = []
    for img_path in image_paths:
        shelf_rows = shelf_det.detect(img_path)
        rows_with_products = prod_det.detect(img_path, shelf_rows)
        rows_with_skus = sku_rec.recognise_rows(img_path, rows_with_products)
        for row in rows_with_skus:
            row.row_idx = len(all_rows)
            all_rows.append(row)

    self.update_state(state="STARTED", meta={"step": "processing"})
    compliance_rows = check_compliance(all_rows, shelf_key)

    overall = (
        sum(r.compliance_score for r in compliance_rows) / len(compliance_rows)
        if compliance_rows else 0.0
    )

    vdir = sess.viz_dir(session_id)
    assets: List[str] = []

    generate_simplified(image_paths, all_rows, vdir / "shelf_analysis_simplified.jpg")
    assets.append(str(vdir / "shelf_analysis_simplified.jpg"))

    generate_results(image_paths, all_rows, vdir / "shelf_analysis_results.jpg")
    assets.append(str(vdir / "shelf_analysis_results.jpg"))

    return {
        "shelf_key": shelf_key,
        "overall_compliance": round(overall, 4),
        "rows": [r.model_dump() for r in compliance_rows],
        "assets": assets,
        "lang": lang,
    }
