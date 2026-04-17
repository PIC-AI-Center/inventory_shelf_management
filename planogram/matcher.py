"""Top-K planogram matching."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from planogram.loader import all_planograms
from schemas import DetectedRow, PlanogramCandidate, RowMatch


def _row_similarity(detected_skus: List[str], planogram_skus: List[str]) -> float:
    a = set(detected_skus)
    b = set(planogram_skus)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _match_rows(
    detected_rows: List[DetectedRow],
    planogram_rows: List[List[str]],
) -> Tuple[float, List[RowMatch]]:
    if not detected_rows or not planogram_rows:
        return 0.0, []

    n_det = len(detected_rows)
    n_plan = len(planogram_rows)
    sim_matrix = np.zeros((n_det, n_plan), dtype=np.float32)
    for i, drow in enumerate(detected_rows):
        det_skus = [p.sku_id or p.label for p in drow.products]
        for j, plan_row in enumerate(planogram_rows):
            sim_matrix[i, j] = _row_similarity(det_skus, plan_row)

    matched: List[RowMatch] = []
    used_plan = set()
    scores = []
    for i in range(n_det):
        best_j = -1
        best_sim = -1.0
        for j in range(n_plan):
            if j not in used_plan and sim_matrix[i, j] > best_sim:
                best_sim = sim_matrix[i, j]
                best_j = j
        if best_j >= 0:
            used_plan.add(best_j)
            matched.append(
                RowMatch(
                    detected_row_idx=i,
                    planogram_row_idx=best_j,
                    similarity=float(best_sim),
                )
            )
            scores.append(float(best_sim))

    overall = float(np.mean(scores)) if scores else 0.0
    return overall, matched


def top_k_match(
    detected_rows: List[DetectedRow],
    top_k: int = 3,
) -> List[PlanogramCandidate]:
    planograms = all_planograms()
    results: List[PlanogramCandidate] = []
    for planogram_id, plan_rows in planograms.items():
        score, row_matches = _match_rows(detected_rows, plan_rows)
        results.append(
            PlanogramCandidate(
                planogram_id=planogram_id,
                score=score,
                matched_rows=row_matches,
            )
        )
    results.sort(key=lambda c: c.score, reverse=True)
    return results[:top_k]
