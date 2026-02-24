"""
metrics/collection.py
=====================
Evaluation metrics for the data collection task.

The collection task is treated as an information retrieval problem:
given a person, did the system retrieve the right URLs?

Metrics
-------
- Per-case precision, recall, F1 over retrieved vs gold URL sets
- Macro-averaged precision, recall, F1 across all cases

Input contract
--------------
Both predictions and gold are passed as dicts keyed by case_id,
with values being sets of URL strings. The caller (evaluate.py)
handles all dataframe alignment before calling these functions.
"""

from __future__ import annotations


def collection_metrics(
    predictions: dict[str, set[str]],
    gold: dict[str, set[str]],
) -> dict:
    """
    Compute precision, recall, and F1 for URL retrieval.

    Parameters
    ----------
    predictions : {case_id: set of retrieved URLs}
    gold        : {case_id: set of gold-standard relevant URLs}
                  Only case_ids present in both dicts are evaluated.

    Returns
    -------
    {
        "macro": {
            "precision": float,
            "recall":    float,
            "f1":        float,
        },
        "per_case": {
            "<case_id>": {
                "precision": float,
                "recall":    float,
                "f1":        float,
                "n_retrieved": int,
                "n_gold":      int,
                "n_correct":   int,
            },
            ...
        },
        "n_cases": int,
        "n_skipped": int,   # case_ids in predictions with no gold entry
    }
    """
    per_case: dict[str, dict] = {}
    skipped: list[str] = []

    for case_id, retrieved in predictions.items():
        relevant = gold.get(str(case_id))

        if relevant is None:
            skipped.append(str(case_id))
            continue

        p, r, f = _precision_recall_f1(retrieved, relevant)
        per_case[str(case_id)] = {
            "precision":   round(p, 6),
            "recall":      round(r, 6),
            "f1":          round(f, 6),
            "n_retrieved": len(retrieved),
            "n_gold":      len(relevant),
            "n_correct":   len(retrieved & relevant),
        }

    n = len(per_case)
    if n == 0:
        macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    else:
        macro = {
            "precision": round(sum(v["precision"] for v in per_case.values()) / n, 6),
            "recall":    round(sum(v["recall"]    for v in per_case.values()) / n, 6),
            "f1":        round(sum(v["f1"]        for v in per_case.values()) / n, 6),
        }

    return {
        "macro":    macro,
        "per_case": per_case,
        "n_cases":  n,
        "n_skipped": len(skipped),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _precision_recall_f1(
    predicted: set[str],
    relevant: set[str],
) -> tuple[float, float, float]:
    """
    Compute precision, recall, F1 for a single case.

    Edge cases:
    - Both empty → (1.0, 1.0, 1.0): system correctly retrieved nothing
    - Predicted empty, gold non-empty → (0.0, 0.0, 0.0)
    - Gold empty, predicted non-empty → (0.0, 0.0, 0.0): no relevant URLs
      exist, anything retrieved is wrong
    """
    if not predicted and not relevant:
        return 1.0, 1.0, 1.0
    if not predicted or not relevant:
        return 0.0, 0.0, 0.0

    tp        = len(predicted & relevant)
    precision = tp / len(predicted)
    recall    = tp / len(relevant)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1