"""Shelf compliance: compare detected rows against a confirmed planogram."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from planogram.loader import load_planogram
from schemas import DetectedRow, ShelfRowCompliance


def check_compliance(
    detected_rows: List[DetectedRow],
    planogram_id: str,
    row_matches: Optional[List[Dict[str, Any]]] = None,
) -> List[ShelfRowCompliance]:
    """Compare detected rows to planogram rows.

    row_matches: optional list of {detected_row_idx, planogram_row_idx} dicts
    from a prior match step.  If absent, rows are aligned positionally.
    """
    planogram_rows = load_planogram(planogram_id)
    results: List[ShelfRowCompliance] = []

    # Build alignment map: detected_row_idx -> planogram_row_idx
    alignment: Dict[int, int] = {}
    if row_matches:
        for m in row_matches:
            alignment[m["detected_row_idx"]] = m["planogram_row_idx"]
    else:
        for i in range(len(detected_rows)):
            if i < len(planogram_rows):
                alignment[i] = i

    for drow in detected_rows:
        plan_idx = alignment.get(drow.row_idx)
        if plan_idx is None or plan_idx >= len(planogram_rows):
            results.append(
                ShelfRowCompliance(
                    row_idx=drow.row_idx,
                    planogram_row_idx=plan_idx,
                    compliance_score=0.0,
                )
            )
            continue

        expected = planogram_rows[plan_idx]
        detected = [p.sku_id or p.label for p in drow.products]

        expected_set = set(expected)
        detected_set = set(detected)
        missing = sorted(expected_set - detected_set)
        unexpected = sorted(detected_set - expected_set)

        if expected_set:
            correct = len(expected_set & detected_set)
            score = correct / len(expected_set)
        else:
            score = 1.0 if not detected_set else 0.0

        results.append(
            ShelfRowCompliance(
                row_idx=drow.row_idx,
                planogram_row_idx=plan_idx,
                expected_skus=expected,
                detected_skus=detected,
                compliance_score=round(score, 4),
                missing=missing,
                unexpected=unexpected,
            )
        )

    return results
