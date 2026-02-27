"""
evaluate.py
===========
Main evaluation function for the CoREx benchmarking library.

This is the second function contributors interact with directly,
after load_inputs(). It takes their predictions dataframe, aligns it
to the gold standard test set, calls the appropriate metric functions,
and returns a results dict.

Typical usage
-------------
    from corex_eval import load_inputs, evaluate

    # 1. Get test inputs
    inputs = load_inputs(task="extraction")

    # 2. Run your model → predictions_df
    # predictions_df must have: [case_id, birth_year]

    # 3. Evaluate
    results = evaluate(
        predictions_df,
        task="extraction",
        variable="birth_year",
    )

    # 4. Optionally submit to shared register
    results = evaluate(
        predictions_df,
        task="extraction",
        variable="birth_year",
        submit=True,
        experiment_path="experiments/extraction/gpt4o/config.yaml",
    )
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from corex_eval.config import (
    ANNOTATION_VARIABLES,
    ATOMIC_VARIABLES,
    CASE_ID_COL,
    COLLECTION_TARGET_COL,
    COMPOSITE_VARIABLES,
    EXTRACTION_VARIABLES,
    SPELL_INDEX_COL,
)

if TYPE_CHECKING:
    import pandas as pd

_VALID_TASKS = {"collection", "extraction", "annotation"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate(
    predictions_df: "pd.DataFrame",
    task: str,
    variable: str | None = None,
    submit: bool = False,
    experiment_path: str | None = None,
    gold_path: str | None = None,
    silver_path: str | None = None,
    semantic_similarity: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
    tolerance_years: int | None = None,
) -> dict:
    """
    Evaluate model predictions against the gold standard test set.

    Parameters
    ----------
    predictions_df  : DataFrame produced by running your model on the
                      output of load_inputs(). Must contain case_id as
                      an alignment key, plus the predicted value column(s).

                      collection → must have: [case_id, retrieved_urls]
                                   retrieved_urls: list[str] or pipe-separated str

                      extraction → must have: [case_id, <variable>]
                                   e.g. [case_id, birth_year] or [case_id, career]
                                   For career: value should be a list of dicts
                                   with keys start_year, end_year, position

                      annotation → must have: [case_id, spell_index, predicted_code]
                                   (spell_index only required for spell-level variables)

    task            : One of "collection", "extraction", "annotation".

    variable        : Required for extraction and annotation.
                      extraction  → any key from ATOMIC_VARIABLES or
                                    COMPOSITE_VARIABLES ("career")
                      annotation  → any key from ANNOTATION_VARIABLES

    submit          : If True, append results to the shared GitHub register.
                      Requires GITHUB_TOKEN env var and experiment_path.

    experiment_path : Path to your experiment config.yaml. Required when
                      submit=True. Used to label the register entry.

    gold_path       : Optional override for gold CSV path.
    silver_path     : Optional override for silver CSV path.

    semantic_similarity : annotation only. If True, compute BERT cosine
                          similarity between predicted and gold labels.
                          Requires pip install corex_eval[embeddings].

    embedding_model : Sentence-transformers model for semantic similarity.

    tolerance_years : Override for temporal matching tolerance in composite
                      extraction. Defaults to TEMPORAL_TOLERANCE_YEARS
                      from config.py.

    Returns
    -------
    Results dict with structure depending on task:

        collection → see metrics.collection.collection_metrics()
        extraction → see metrics.extraction.atomic_accuracy() /
                     atomic_mae() / composite_metrics()
        annotation → see metrics.annotation.annotation_metrics()

    Always includes top-level keys:
        "task", "variable", "n_evaluated", "n_skipped"
    """
    # --- Validate inputs ---
    _validate_task(task)
    _validate_variable(task, variable)
    _validate_predictions_df(predictions_df)

    if submit and not experiment_path:
        raise ValueError(
            "experiment_path is required when submit=True. "
            "Pass the path to your experiment config.yaml."
        )

    # --- Load gold and apply test split ---
    from corex_eval.gold import load_gold
    from corex_eval.split import apply_test_split

    gold_df  = load_gold(gold_path)
    test_df  = apply_test_split(gold_df)

    # --- Route to task-specific evaluator ---
    if task == "collection":
        results = _evaluate_collection(predictions_df, test_df)

    elif task == "extraction":
        results = _evaluate_extraction(
            predictions_df, test_df, variable, tolerance_years
        )

    elif task == "annotation":
        from corex_eval.silver import load_silver
        silver_df = load_silver(silver_path)
        gold_case_ids = set(gold_df[CASE_ID_COL].astype(str))
        results = _evaluate_annotation(
            predictions_df, test_df, silver_df,
            variable, gold_case_ids,
            semantic_similarity, embedding_model,
        )

    # --- Add task metadata ---
    results["task"]     = task
    results["variable"] = variable

    # --- Optionally submit ---
    if submit:
        from corex_eval.submit import submit_results
        submit_results(results, experiment_path=experiment_path)

    _print_summary(results, task, variable)
    return results


# ---------------------------------------------------------------------------
# Task-specific evaluators
# ---------------------------------------------------------------------------

def _evaluate_collection(
    predictions_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
) -> dict:
    """
    Align predictions to gold and compute collection metrics.

    predictions_df must have: [case_id, retrieved_urls]
    retrieved_urls can be a list[str] or a pipe-separated string.
    """
    from corex_eval.metrics.collection import collection_metrics

    _require_columns(predictions_df, [CASE_ID_COL, "retrieved_urls"])

    # Build gold dict: {case_id: set of URLs}
    gold_dict: dict[str, set[str]] = {}
    for _, row in test_df.iterrows():
        case_id = str(row[CASE_ID_COL])
        raw     = str(row.get(COLLECTION_TARGET_COL, "")).strip()
        urls    = _parse_urls(raw)
        if urls:
            gold_dict[case_id] = urls

    # Build predictions dict: {case_id: set of URLs}
    pred_dict: dict[str, set[str]] = {}
    for _, row in predictions_df.iterrows():
        case_id = str(row[CASE_ID_COL])
        raw     = row.get("retrieved_urls", "")
        if isinstance(raw, list):
            urls = set(str(u).strip() for u in raw if str(u).strip())
        else:
            urls = _parse_urls(str(raw))
        pred_dict[case_id] = urls

    results = collection_metrics(pred_dict, gold_dict)
    results["n_evaluated"] = results.pop("n_cases")
    results["n_skipped"]   = results.pop("n_skipped")
    return results


def _evaluate_extraction(
    predictions_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    variable: str,
    tolerance_years: int | None,
) -> dict:
    """
    Route to atomic or composite extraction metrics depending on variable.
    """
    if variable in ATOMIC_VARIABLES:
        return _evaluate_atomic(predictions_df, test_df, variable)
    else:
        return _evaluate_composite(predictions_df, test_df, variable, tolerance_years)


def _evaluate_atomic(
    predictions_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    variable: str,
) -> dict:
    """
    Evaluate an atomic extraction variable (single value per case).

    Aligns predictions to test set by case_id, then calls either
    atomic_accuracy (str) or atomic_mae (int) depending on variable type.
    """
    from corex_eval.metrics.extraction import atomic_accuracy, atomic_mae

    col, dtype = ATOMIC_VARIABLES[variable]
    _require_columns(predictions_df, [CASE_ID_COL, variable])

    # Align: keep only test case_ids, in test_df order
    pred_index = (
        predictions_df
        .set_index(CASE_ID_COL)[variable]
        .to_dict()
    )

    aligned_pred = []
    aligned_gold = []
    missing_preds = []

    for _, row in test_df.iterrows():
        case_id  = str(row[CASE_ID_COL])
        gold_val = row.get(col)

        if case_id not in pred_index:
            missing_preds.append(case_id)
            continue

        aligned_pred.append(pred_index[case_id])
        aligned_gold.append(gold_val)

    if missing_preds:
        warnings.warn(
            f"[evaluate] {len(missing_preds)} test case_id(s) have no prediction "
            f"and are excluded from scoring: {missing_preds[:10]}"
            f"{'...' if len(missing_preds) > 10 else ''}"
        )

    if dtype == "int":
        results = atomic_mae(aligned_pred, aligned_gold)
    else:
        results = atomic_accuracy(aligned_pred, aligned_gold)

    results["n_evaluated"] = results.pop("n_total")
    results["n_skipped"]   = results.pop("n_skipped") + len(missing_preds)
    return results


def _evaluate_composite(
    predictions_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    variable: str,
    tolerance_years: int | None,
) -> dict:
    """
    Evaluate composite extraction (career history).

    predictions_df must have [case_id, career] where career is a list
    of dicts with keys: start_year, end_year, position (all optional).
    """
    from corex_eval.metrics.extraction import composite_metrics, CareerSpell
    from corex_eval.gold import get_career_spells

    _require_columns(predictions_df, [CASE_ID_COL, variable])

    tol = tolerance_years if tolerance_years is not None else (
        __import__("corex_eval.config", fromlist=["TEMPORAL_TOLERANCE_YEARS"])
        .TEMPORAL_TOLERANCE_YEARS
    )

    # Build gold spell dict from reshaped long format
    career_long = get_career_spells(test_df)
    gold_dict: dict[str, list[CareerSpell]] = {}
    for _, row in career_long.iterrows():
        case_id = str(row[CASE_ID_COL])
        spell   = CareerSpell(
            case_id    = case_id,
            start_year = _to_int(row.get("career_start_year")),
            end_year   = _to_int(row.get("career_end_year")),
            position   = str(row.get("career_position", "")) or None,
        )
        if spell.start_year is None and spell.end_year is None:
            continue  # skip spells with no temporal info — unevaluable
        gold_dict.setdefault(case_id, []).append(spell)

    # Build predictions spell dict
    pred_dict: dict[str, list[CareerSpell]] = {}
    missing_preds = []

    for _, row in predictions_df.iterrows():
        case_id = str(row[CASE_ID_COL])
        raw     = row.get(variable, [])

        if not isinstance(raw, list):
            warnings.warn(
                f"[evaluate] case_id={case_id}: expected a list of dicts for "
                f"'{variable}', got {type(raw).__name__}. Skipping."
            )
            missing_preds.append(case_id)
            continue

        spells = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            spell = CareerSpell(
                case_id=case_id,
                start_year=_to_int(entry.get("start_year")),
                end_year=_to_int(entry.get("end_year")),
                position=entry.get("position"),
            )
            if spell.start_year is None and spell.end_year is None:
                continue  # unevaluable — skip on both sides
            spells.append(spell)
        pred_dict[case_id] = spells

    results = composite_metrics(pred_dict, gold_dict, tolerance_years=tol)
    results["n_evaluated"] = results.pop("n_cases")
    results["n_skipped"]   = results.pop("n_skipped") + len(missing_preds)
    results["tolerance_years"] = tol
    return results


def _evaluate_annotation(
    predictions_df: "pd.DataFrame",
    test_df: "pd.DataFrame",
    silver_df: "pd.DataFrame",
    variable: str,
    gold_case_ids: "set[str]",
    semantic_similarity: bool,
    embedding_model: str,
) -> dict:
    """
    Align predictions to gold annotation codes and compute classification metrics.

    Alignment key:
      - (case_id, spell_index)  for spell-level variables (career_position, uni_subject)
      - case_id alone           for atomic variables

    For all spell-level variables, the gold code is stored inline in each silver
    row (career_position / uni_subject columns) so no wide-format reshaping is needed.
    """
    from corex_eval.metrics.annotation import annotation_metrics
    from corex_eval.silver import get_silver_inputs

    var_config  = ANNOTATION_VARIABLES[variable]
    gold_col    = var_config["gold_col"]
    has_spells  = var_config["spell_index_col"] is not None

    test_case_ids = set(test_df[CASE_ID_COL].astype(str))

    # Determine required prediction columns
    if has_spells:
        _require_columns(predictions_df, [CASE_ID_COL, SPELL_INDEX_COL, "predicted_code"])
    else:
        _require_columns(predictions_df, [CASE_ID_COL, "predicted_code"])

    # Build gold dict
    gold_dict: dict = {}
    if has_spells:
        # Gold code is stored inline in the silver rows.
        # Alignment key: (case_id, spell_index) → gold_code.
        silver_inputs = get_silver_inputs(silver_df, variable, gold_case_ids)
        for _, row in silver_inputs.iterrows():
            if str(row[CASE_ID_COL]) not in test_case_ids:
                continue
            key = (str(row[CASE_ID_COL]), int(row[SPELL_INDEX_COL]))
            gold_dict[key] = str(row.get(gold_col, "")).strip()
    else:
        # Gold is atomic — {case_id: gold_code}
        gold_dict = {
            str(row[CASE_ID_COL]): str(row.get(gold_col, "")).strip()
            for _, row in test_df.iterrows()
        }

    # Align predictions to gold
    aligned_pred = []
    aligned_gold = []
    missing_preds = []

    for _, row in predictions_df.iterrows():
        case_id = str(row[CASE_ID_COL])
        if case_id not in test_case_ids:
            continue  # skip non-test rows silently

        pred_code = str(row.get("predicted_code", "")).strip()

        if has_spells:
            spell_idx = row.get(SPELL_INDEX_COL)
            if spell_idx is None:
                missing_preds.append(case_id)
                continue
            key = (case_id, int(spell_idx))
            gold_code = gold_dict.get(key)
        else:
            gold_code = gold_dict.get(case_id)

        if gold_code is None or not gold_code:
            missing_preds.append(case_id)
            continue

        aligned_pred.append(pred_code)
        aligned_gold.append(gold_code)

    if missing_preds:
        warnings.warn(
            f"[evaluate] {len(missing_preds)} prediction(s) could not be aligned "
            f"to a gold annotation entry and were excluded: "
            f"{missing_preds[:10]}"
            f"{'...' if len(missing_preds) > 10 else ''}"
        )

    results = annotation_metrics(
        aligned_pred,
        aligned_gold,
        semantic_similarity=semantic_similarity,
        embedding_model=embedding_model,
    )

    results["n_evaluated"] = results.pop("n_total")
    results["n_skipped"]   = len(missing_preds)
    return results


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def _print_summary(results: dict, task: str, variable: str | None) -> None:
    """Print a human-readable summary of the results to stdout."""
    label = f"{task}" + (f" / {variable}" if variable else "")
    print(f"\n{'─' * 50}")
    print(f"  CoREx evaluation — {label}")
    print(f"{'─' * 50}")
    print(f"  Cases evaluated : {results.get('n_evaluated', '?')}")
    print(f"  Cases skipped   : {results.get('n_skipped',   '?')}")

    if task == "collection":
        m = results.get("macro", {})
        print(f"  Precision (macro): {m.get('precision', '?'):.4f}")
        print(f"  Recall    (macro): {m.get('recall',    '?'):.4f}")
        print(f"  F1        (macro): {m.get('f1',        '?'):.4f}")

    elif task == "extraction":
        if "mae" in results:
            print(f"  MAE      : {results.get('mae',      '?')}")
            print(f"  Accuracy : {results.get('accuracy', '?')}")
        elif "macro" in results:
            m = results.get("macro", {})
            print(f"  Precision (macro): {m.get('precision', '?'):.4f}")
            print(f"  Recall    (macro): {m.get('recall',    '?'):.4f}")
            print(f"  F1        (macro): {m.get('f1',        '?'):.4f}")
        else:
            print(f"  Accuracy : {results.get('accuracy', '?')}")

    elif task == "annotation":
        print(f"  Accuracy     : {results.get('accuracy',    '?')}")
        print(f"  Macro F1     : {results.get('macro_f1',    '?')}")
        print(f"  Weighted F1  : {results.get('weighted_f1', '?')}")
        if results.get("semantic_similarity") is not None:
            print(f"  Semantic sim : {results['semantic_similarity']:.4f}")

    print(f"{'─' * 50}\n")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_urls(raw: str) -> set[str]:
    """Parse a pipe-separated URL string into a set of stripped URL strings."""
    return set(u.strip() for u in raw.split("|") if u.strip())


def _to_int(value) -> int | None:
    """Convert a value to int, returning None if not possible."""
    if value is None:
        return None
    try:
        v = int(value)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _require_columns(df: "pd.DataFrame", cols: list[str]) -> None:
    """Raise a clear error if any required columns are missing."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"predictions_df is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_task(task: str) -> None:
    if task not in _VALID_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Valid options: {sorted(_VALID_TASKS)}"
        )


def _validate_variable(task: str, variable: str | None) -> None:
    if task == "collection":
        return  # variable not used for collection
    if variable is None:
        raise ValueError(
            f"'variable' is required for task='{task}'."
        )
    if task == "extraction" and variable not in EXTRACTION_VARIABLES:
        raise ValueError(
            f"Unknown extraction variable '{variable}'. "
            f"Valid options: {sorted(EXTRACTION_VARIABLES)}"
        )
    if task == "annotation" and variable not in ANNOTATION_VARIABLES:
        raise ValueError(
            f"Unknown annotation variable '{variable}'. "
            f"Valid options: {sorted(ANNOTATION_VARIABLES)}"
        )


def _validate_predictions_df(df: "pd.DataFrame") -> None:
    if not hasattr(df, "columns"):
        raise TypeError(
            "predictions_df must be a pandas DataFrame."
        )
    if CASE_ID_COL not in df.columns:
        raise ValueError(
            f"predictions_df must contain a '{CASE_ID_COL}' column for alignment."
        )
    if len(df) == 0:
        raise ValueError("predictions_df is empty.")