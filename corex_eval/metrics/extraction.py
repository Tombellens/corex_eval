"""
metrics/extraction.py
=====================
Evaluation metrics for the data extraction task.

The extraction task is split into two types of attributes:

Atomic attributes (single-value fields per person)
    - String fields (birth_place, sex, edu_degree, ...):
      exact-match accuracy after normalisation
    - Integer/year fields (birth_year, edu_start, edu_end):
      exact-match accuracy + mean absolute error (MAE)

Composite attributes (multi-record fields: career history)
    - Record-level precision, recall, F1
    - Records are matched using temporal fuzzy matching with a
      configurable year tolerance (default: 1 year)
    - Macro-averaged across all cases

Input contract
--------------
All functions receive plain Python types — no pandas. The caller
(evaluate.py) handles dataframe alignment and passes lists/dicts here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from corex_eval.config import TEMPORAL_TOLERANCE_YEARS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CareerSpell:
    """A single career position with temporal boundaries."""
    case_id:    str
    start_year: int | None
    end_year:   int | None
    position:   str | None   # raw position string, used for soft matching


# ---------------------------------------------------------------------------
# Atomic metrics
# ---------------------------------------------------------------------------

def atomic_accuracy(
    predictions: list[str | None],
    gold: list[str | None],
) -> dict:
    """
    Exact-match accuracy for a single atomic string variable.

    Comparison is case-insensitive and strips leading/trailing whitespace.
    Pairs where the gold value is None/empty are skipped (no gold to
    compare against).

    Parameters
    ----------
    predictions : List of predicted string values, aligned with gold.
    gold        : List of gold string values.

    Returns
    -------
    {
        "accuracy":  float,   # fraction of correct predictions
        "n_correct": int,
        "n_total":   int,     # pairs where gold is non-null
        "n_skipped": int,     # pairs where gold is null/empty
    }
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"predictions and gold must be the same length "
            f"(got {len(predictions)} vs {len(gold)})"
        )

    n_correct = 0
    n_total   = 0
    n_skipped = 0

    for pred, g in zip(predictions, gold):
        if _is_empty(g):
            n_skipped += 1
            continue
        n_total += 1
        if _normalise(pred) == _normalise(g):
            n_correct += 1

    accuracy = n_correct / n_total if n_total > 0 else None

    return {
        "accuracy":  round(accuracy, 6) if accuracy is not None else None,
        "n_correct": n_correct,
        "n_total":   n_total,
        "n_skipped": n_skipped,
    }


def atomic_mae(
    predictions: list[int | float | None],
    gold: list[int | float | None],
) -> dict:
    """
    Mean absolute error for a single atomic integer/year variable.

    Pairs where either prediction or gold is None are skipped.

    Parameters
    ----------
    predictions : List of predicted numeric values.
    gold        : List of gold numeric values.

    Returns
    -------
    {
        "mae":       float | None,   # None if no valid pairs
        "accuracy":  float | None,   # exact-match rate (useful for years)
        "n_total":   int,
        "n_skipped": int,
    }
    """
    if len(predictions) != len(gold):
        raise ValueError(
            f"predictions and gold must be the same length "
            f"(got {len(predictions)} vs {len(gold)})"
        )

    errors    = []
    n_exact   = 0
    n_skipped = 0

    for pred, g in zip(predictions, gold):
        if _is_empty(g) or _is_empty(pred):
            n_skipped += 1
            continue
        try:
            p_val = float(pred)
            g_val = float(g)
        except (TypeError, ValueError):
            n_skipped += 1
            continue

        errors.append(abs(p_val - g_val))
        if p_val == g_val:
            n_exact += 1

    n_total  = len(errors)
    mae      = sum(errors) / n_total if n_total > 0 else None
    accuracy = n_exact / n_total     if n_total > 0 else None

    return {
        "mae":       round(mae,      6) if mae      is not None else None,
        "accuracy":  round(accuracy, 6) if accuracy is not None else None,
        "n_total":   n_total,
        "n_skipped": n_skipped,
    }


# ---------------------------------------------------------------------------
# Composite metrics (career history)
# ---------------------------------------------------------------------------

def composite_metrics(
    predictions: dict[str, list[CareerSpell]],
    gold: dict[str, list[CareerSpell]],
    tolerance_years: int = TEMPORAL_TOLERANCE_YEARS,
) -> dict:
    """
    Record-level precision, recall, F1 for career history extraction.

    Each case has a list of predicted spells and a list of gold spells.
    Spells are matched using temporal fuzzy matching (see _spells_match).
    Metrics are macro-averaged across all cases.

    Parameters
    ----------
    predictions     : {case_id: list of CareerSpell}
    gold            : {case_id: list of CareerSpell}
    tolerance_years : Maximum year difference still counted as a match.
                      Defaults to TEMPORAL_TOLERANCE_YEARS from config.py.

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
                "precision":       float,
                "recall":          float,
                "f1":              float,
                "n_predicted":     int,
                "n_gold":          int,
                "n_matched":       int,
            },
            ...
        },
        "n_cases":   int,
        "n_skipped": int,
    }
    """
    per_case: dict[str, dict] = {}
    skipped: list[str] = []

    for case_id, pred_spells in predictions.items():
        gold_spells = gold.get(str(case_id))

        if gold_spells is None:
            skipped.append(str(case_id))
            continue

        p, r, f, n_matched = _match_spells(pred_spells, gold_spells, tolerance_years)

        per_case[str(case_id)] = {
            "precision":   round(p, 6),
            "recall":      round(r, 6),
            "f1":          round(f, 6),
            "n_predicted": len(pred_spells),
            "n_gold":      len(gold_spells),
            "n_matched":   n_matched,
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
        "macro":     macro,
        "per_case":  per_case,
        "n_cases":   n,
        "n_skipped": len(skipped),
    }


# ---------------------------------------------------------------------------
# Temporal spell matching
# ---------------------------------------------------------------------------

def _match_spells(
    predicted: list[CareerSpell],
    gold: list[CareerSpell],
    tolerance: int,
) -> tuple[float, float, float, int]:
    """
    Greedy bipartite matching between predicted and gold career spells.

    A predicted spell matches a gold spell if:
      1. start_year values are within `tolerance` years of each other
         (or either is None — treated as unknown / don't care)
      2. end_year values are within `tolerance` years of each other
         (or either is None)

    We do not match on position strings at this stage — extraction is
    evaluated purely on temporal structure, not on label correctness
    (that is the annotation task).

    Returns (precision, recall, f1, n_matched).
    """
    if not predicted and not gold:
        return 1.0, 1.0, 1.0, 0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0, 0

    matched_gold: set[int] = set()
    n_matched = 0

    for pred_spell in predicted:
        best_idx = None
        for i, gold_spell in enumerate(gold):
            if i in matched_gold:
                continue
            if _spells_match(pred_spell, gold_spell, tolerance):
                best_idx = i
                break   # take first match (greedy)
        if best_idx is not None:
            matched_gold.add(best_idx)
            n_matched += 1

    precision = n_matched / len(predicted)
    recall    = n_matched / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1, n_matched


def _spells_match(
    a: CareerSpell,
    b: CareerSpell,
    tolerance: int,
) -> bool:
    """
    Return True if two career spells are temporally compatible.

    Rules:
    - If both start_years are present, they must be within tolerance.
    - If both end_years are present, they must be within tolerance.
    - None on either side of a comparison means "unknown" → always compatible.
    - At least one of start_year or end_year must be present in the
      predicted spell (a spell with no temporal info matches nothing).
    """
    # Reject spells with no temporal information at all
    if a.start_year is None and a.end_year is None:
        return False

    return (
        _years_compatible(a.start_year, b.start_year, tolerance)
        and _years_compatible(a.end_year, b.end_year, tolerance)
    )


def _years_compatible(
    a: int | None,
    b: int | None,
    tolerance: int,
) -> bool:
    """Two years are compatible if either is None, or they're within tolerance."""
    if a is None or b is None:
        return True
    return abs(a - b) <= tolerance


# ---------------------------------------------------------------------------
# String normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(value: str | None) -> str:
    """Lowercase and strip for case-insensitive comparison."""
    if value is None:
        return ""
    return str(value).strip().lower()


def _is_empty(value) -> bool:
    """Return True for None, empty string, 'nan', '<NA>', '/'."""
    if value is None:
        return True
    s = str(value).strip()
    return s in ("", "nan", "<NA>", "/", "99")