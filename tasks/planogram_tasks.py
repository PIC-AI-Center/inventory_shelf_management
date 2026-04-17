from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from celery_app import celery_app
from config import get_settings
from inference.shelf_detector import ShelfDetector
from inference.product_detector import ProductDetector
from inference.sku_recognizer import SKURecognizer
from planogram.matcher import top_k_match
from planogram.compliance import check_compliance
from planogram.loader import load_planogram
from sessions import manager as sess
from visualization.simplified import generate_simplified
from visualization.results import generate_results
from visualization.detection import generate_detection_viz
from schemas import DetectedRow


@celery_app.task(name="tasks.planogram_tasks.match_planogram", bind=True)
def match_planogram(
    self,
    match_id: str,
    b64_images: List[str],
    image_orientations: List[int],
    top_k: int,
    store_id: str,
) -> Dict[str, Any]:
    settings = get_settings()
    self.update_state(state="STARTED", meta={"step": "saving_images"})

    image_paths = sess.save_images(match_id, b64_images)

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

    self.update_state(state="STARTED", meta={"step": "matching"})
    candidates = top_k_match(all_rows, top_k=top_k)

    total_labels = sum(len(r.products) for r in all_rows)
    meta = {
        "num_images": len(image_paths),
        "num_detected_rows": len(all_rows),
        "total_labels": total_labels,
        "store_id": store_id,
    }
    sess.save_meta(match_id, meta)
    sess.save_detected(match_id, {
        "rows": [r.model_dump() for r in all_rows],
        "image_paths": [str(p) for p in image_paths],
    })

    return {
        "match_id": match_id,
        "top_k": top_k,
        "matches": [c.model_dump() for c in candidates],
        "meta": meta,
    }


@celery_app.task(name="tasks.planogram_tasks.finalize_planogram_match", bind=True)
def finalize_planogram_match(
    self,
    match_id: str,
    planogram_id: str,
    lang: str,
    create_visualizations: bool,
    delete_after: bool,
) -> Dict[str, Any]:
    self.update_state(state="STARTED", meta={"step": "loading"})

    detected_data = sess.load_detected(match_id)
    all_rows = [DetectedRow(**r) for r in detected_data["rows"]]
    image_paths = [Path(p) for p in detected_data["image_paths"]]

    self.update_state(state="STARTED", meta={"step": "processing"})
    plan_rows = load_planogram(planogram_id)
    compliance_rows = check_compliance(all_rows, planogram_id)

    result: Dict[str, Any] = {
        "shelf_key": planogram_id,
        "rows": [r.model_dump() for r in compliance_rows],
        "overall_compliance": (
            sum(r.compliance_score for r in compliance_rows) / len(compliance_rows)
            if compliance_rows else 0.0
        ),
        "lang": lang,
    }

    vdir = sess.viz_dir(match_id)
    assets: List[str] = []

    self.update_state(state="STARTED", meta={"step": "rendering"})

    simplified_path = vdir / "shelf_analysis_simplified.jpg"
    generate_simplified(image_paths, all_rows, simplified_path)
    assets.append("shelf_analysis_simplified.jpg")

    results_path = vdir / "shelf_analysis_results.jpg"
    generate_results(image_paths, all_rows, results_path)
    assets.append("shelf_analysis_results.jpg")

    if create_visualizations:
        for idx, img_path in enumerate(image_paths):
            if img_path.exists():
                out = generate_detection_viz(img_path, idx, all_rows, vdir)
                assets.append(out.name)

    if delete_after:
        sess.delete_session(match_id)

    return {
        "match_id": match_id,
        "planogram_id": planogram_id,
        "result": result,
        "meta": {
            "rows_in_planogram": len(plan_rows),
            "assets": assets,
        },
    }
