"""
inputs.py
=========
Public functions for loading model inputs from the CoREx dataset.

This is the first module contributors interact with directly. It provides:

  load_inputs(task, variable=None)
      Returns the 20% test set inputs for a given task. Contributors
      run their model on this and produce a predictions dataframe.

  load_training_data(task, features, variable=None)
      Returns the 80% training set with the requested columns. Used
      for fine-tuning or few-shot prompt construction.

Both functions return a pandas DataFrame with case_id as the first column,
so predictions can always be aligned back to the gold standard.


Typical usage
-------------
    from corex_eval import load_inputs, load_training_data

    # Get test set inputs for extraction
    inputs = load_inputs(task="extraction")
    # → DataFrame: [case_id, cv_local]

    # Get test set inputs for annotation
    inputs = load_inputs(task="annotation", variable="career_position")
    # → DataFrame: [case_id, spell_index, job_description_label]

    # Get training data for extraction with selected features
    train = load_training_data(
        task="extraction",
        features=["cv_local", "birth_year", "sex"]
    )
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from corex_eval.config import (
    ANNOTATION_VARIABLES,
    ATOMIC_VARIABLES,
    CASE_ID_COL,
    COLLECTION_INPUT_COLS,
    COMPOSITE_VARIABLES,
    EXTRACTION_INPUT_COLS,
    EXTRACTION_TEXT_COL,
    SPELL_INDEX_COL,
)

if TYPE_CHECKING:
    import pandas as pd

# Valid task names
_VALID_TASKS = {"collection", "extraction", "annotation"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_inputs(
    task: str,
    variable: str | None = None,
    gold_path: str | None = None,
    silver_path: str | None = None,
) -> "pd.DataFrame":
    """
    Load the 20% test set inputs for a given pipeline task.

    Contributors run their model on the returned dataframe and produce
    a predictions dataframe with the same case_id (and spell_index for
    annotation) for submission to evaluate().

    Parameters
    ----------
    task     : One of "collection", "extraction", "annotation".
    variable : Required for "annotation" (e.g. "career_position",
               "edu_degree", "uni_subject").
               Optional for "extraction" — if provided, the gold column
               for that variable is NOT included (inputs only, no labels).
               Ignored for "collection".
    gold_path   : Optional path override for the gold CSV.
    silver_path : Optional path override for the silver CSV.

    Returns
    -------
    DataFrame containing the test set input columns for the given task.

        collection → [case_id, name_first, name_last, job_title, country_label]
        extraction → [case_id, cv_local]
                     (rows where cv_local is empty are dropped with a warning)
        annotation → [case_id, spell_index, <silver_input_col>]
                     (test set rows from silver standard only)

    Notes
    -----
    - Row order is sorted by case_id for reproducibility.
    - The returned dataframe never contains gold labels — only inputs.
    """
    _validate_task(task)
    gold_df = _load_clean_gold(gold_path)
    test_df = _apply_test_split(gold_df)

    if task == "collection":
        return _collection_inputs(test_df)

    elif task == "extraction":
        return _extraction_inputs(test_df)

    elif task == "annotation":
        _validate_annotation_variable(variable)
        silver_df = _load_clean_silver(silver_path)
        gold_case_ids = set(gold_df[CASE_ID_COL].astype(str))
        test_case_ids = set(test_df[CASE_ID_COL].astype(str))
        return _annotation_inputs(silver_df, variable, gold_case_ids, test_case_ids)


def load_training_data(
    task: str,
    features: list[str],
    variable: str | None = None,
    gold_path: str | None = None,
    silver_path: str | None = None,
) -> "pd.DataFrame":
    """
    Load the 80% training set with the requested feature columns.

    Contributors use this to build training datasets for fine-tuning
    or to construct few-shot examples for prompting.

    Parameters
    ----------
    task     : One of "collection", "extraction", "annotation".
    features : List of column names to include in the returned dataframe.
               case_id is always included as the first column, even if
               not listed in features.
               For extraction, valid columns include "cv_local" plus any
               gold column name (e.g. "birth_year", "career_position_1").
               For annotation, include silver label columns plus gold
               code columns if needed for supervised training.
    variable : Required for "annotation". Determines which silver rows
               to include (career_position, edu_degree, or uni_subject).
    gold_path   : Optional path override for the gold CSV.
    silver_path : Optional path override for the silver CSV.

    Returns
    -------
    DataFrame with [case_id] + requested feature columns, 80% split rows.
    Columns not found in the dataset are skipped with a warning.

    Example
    -------
    >>> train = load_training_data(
    ...     task="extraction",
    ...     features=["cv_local", "birth_year", "sex", "edu_degree"]
    ... )
    >>> train.columns
    Index(['case_id', 'cv_local', 'birth_year', 'sex', 'edu_degree'], dtype='object')
    """
    _validate_task(task)
    gold_df  = _load_clean_gold(gold_path)
    train_df = _apply_train_split(gold_df)

    if task == "annotation":
        _validate_annotation_variable(variable)
        silver_df     = _load_clean_silver(silver_path)
        gold_case_ids = set(gold_df[CASE_ID_COL].astype(str))
        train_case_ids = set(train_df[CASE_ID_COL].astype(str))
        base_df = _annotation_inputs(
            silver_df, variable, gold_case_ids, train_case_ids
        )
    else:
        base_df = train_df

    return _select_features(base_df, features)


# ---------------------------------------------------------------------------
# Task-specific input builders
# ---------------------------------------------------------------------------

def _collection_inputs(df: "pd.DataFrame") -> "pd.DataFrame":
    """Return name + position columns for the collection task."""
    available = [c for c in COLLECTION_INPUT_COLS if c in df.columns]
    missing   = [c for c in COLLECTION_INPUT_COLS if c not in df.columns]
    if missing:
        warnings.warn(
            f"[inputs] Collection input columns not found in dataset: {missing}. "
            f"They will be absent from the returned dataframe."
        )
    return df[available].sort_values(CASE_ID_COL).reset_index(drop=True)


def _extraction_inputs(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Return [case_id, cv_local] for extraction, dropping rows where
    cv_local is empty.
    """
    cols = [c for c in EXTRACTION_INPUT_COLS if c in df.columns]
    result = df[cols].copy()

    # Drop rows where cv_local is missing or empty
    if EXTRACTION_TEXT_COL in result.columns:
        empty_mask = (
            result[EXTRACTION_TEXT_COL]
            .astype(str)
            .str.strip()
            .isin(["", "nan", "<NA>"])
        )
        n_dropped = empty_mask.sum()
        if n_dropped:
            warnings.warn(
                f"[inputs] Dropped {n_dropped} test row(s) where "
                f"'{EXTRACTION_TEXT_COL}' is empty. These rows have no "
                f"text for the model to work with."
            )
        result = result[~empty_mask]

    return result.sort_values(CASE_ID_COL).reset_index(drop=True)


def _annotation_inputs(
    silver_df: "pd.DataFrame",
    variable: str,
    gold_case_ids: "set[str]",
    target_case_ids: "set[str]",
) -> "pd.DataFrame":
    """
    Return silver standard rows for the given variable, filtered to
    target_case_ids (either test or train split).
    """
    from corex_eval.silver import get_silver_inputs

    # get_silver_inputs aligns to gold_case_ids (removes silver rows
    # with no matching gold record)
    silver_inputs = get_silver_inputs(silver_df, variable, gold_case_ids)

    # Further restrict to the target split (test or train)
    result = silver_inputs[
        silver_inputs[CASE_ID_COL].astype(str).isin(target_case_ids)
    ]

    n_dropped = len(silver_inputs) - len(result)
    if n_dropped:
        # This is expected — just the other split's rows being filtered out
        pass

    if SPELL_INDEX_COL in result.columns:
        sort_cols = [CASE_ID_COL, SPELL_INDEX_COL]
    else:
        sort_cols = [CASE_ID_COL]

    return result.sort_values(sort_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature selection helper
# ---------------------------------------------------------------------------

def _select_features(df: "pd.DataFrame", features: list[str]) -> "pd.DataFrame":
    """
    Select requested feature columns from df.

    case_id is always included as the first column. Requested columns
    not present in df are skipped with a warning.
    """
    # Always include case_id first
    requested = [CASE_ID_COL] + [f for f in features if f != CASE_ID_COL]
    available = [c for c in requested if c in df.columns]
    missing   = [c for c in requested if c not in df.columns and c != CASE_ID_COL]

    if missing:
        warnings.warn(
            f"[inputs] Requested feature(s) not found in dataset: {missing}. "
            f"They will be absent from the returned dataframe. "
            f"Check column names against the dataset schema."
        )

    return df[available].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cached loaders (avoid re-reading files on repeated calls)
# ---------------------------------------------------------------------------

def _load_clean_gold(path: str | None) -> "pd.DataFrame":
    from corex_eval.gold import load_gold
    return load_gold(path)


def _load_clean_silver(path: str | None) -> "pd.DataFrame":
    from corex_eval.silver import load_silver
    return load_silver(path)


def _apply_test_split(df: "pd.DataFrame") -> "pd.DataFrame":
    from corex_eval.split import apply_test_split
    return apply_test_split(df)


def _apply_train_split(df: "pd.DataFrame") -> "pd.DataFrame":
    from corex_eval.split import apply_train_split
    return apply_train_split(df)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_task(task: str) -> None:
    if task not in _VALID_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. "
            f"Valid options: {sorted(_VALID_TASKS)}"
        )


def _validate_annotation_variable(variable: str | None) -> None:
    if variable is None:
        raise ValueError(
            "The 'variable' argument is required for task='annotation'. "
            f"Valid options: {sorted(ANNOTATION_VARIABLES)}"
        )
    if variable not in ANNOTATION_VARIABLES:
        raise ValueError(
            f"Unknown annotation variable '{variable}'. "
            f"Valid options: {sorted(ANNOTATION_VARIABLES)}"
        )