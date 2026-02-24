"""
split.py
========
Deterministic 80/20 train/test split for the CoREx gold standard dataset.

This module is internal — contributors never call it directly. It is used
by gold.py, inputs.py, and evaluate.py to ensure everyone works with the
exact same test set.

The split is:
  - Stratified by country_label, so every country is proportionally
    represented in both train and test sets.
  - Seeded with SPLIT_SEED from config.py — this never changes once set.
  - Computed on the *cleaned* gold dataframe (incomplete records already
    removed before splitting).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from corex_eval.config import (
    CASE_ID_COL,
    SPLIT_SEED,
    SPLIT_TEST_SIZE,
    STRATIFY_COL,
)

if TYPE_CHECKING:
    import pandas as pd


def get_test_ids(df: "pd.DataFrame") -> set[str]:
    """
    Return the set of case_ids assigned to the test split.

    Parameters
    ----------
    df : The cleaned gold dataframe (incomplete records already removed).
         Must contain CASE_ID_COL and STRATIFY_COL columns.

    Returns
    -------
    Set of case_id strings belonging to the 20% test set.
    """
    train_ids, test_ids = _split(df)
    return set(test_ids)


def get_train_ids(df: "pd.DataFrame") -> set[str]:
    """
    Return the set of case_ids assigned to the training split (80%).

    Parameters
    ----------
    df : The cleaned gold dataframe (incomplete records already removed).

    Returns
    -------
    Set of case_id strings belonging to the 80% training set.
    """
    train_ids, test_ids = _split(df)
    return set(train_ids)


def apply_test_split(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Filter df to test set rows only.

    Parameters
    ----------
    df : The cleaned gold dataframe.

    Returns
    -------
    DataFrame containing only the 20% test rows, index reset.
    """
    test_ids = get_test_ids(df)
    return df[df[CASE_ID_COL].astype(str).isin(test_ids)].reset_index(drop=True)


def apply_train_split(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Filter df to training set rows only.

    Parameters
    ----------
    df : The cleaned gold dataframe.

    Returns
    -------
    DataFrame containing only the 80% training rows, index reset.
    """
    train_ids = get_train_ids(df)
    return df[df[CASE_ID_COL].astype(str).isin(train_ids)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _split(df: "pd.DataFrame") -> tuple[list[str], list[str]]:
    """
    Core split logic. Returns (train_ids, test_ids) as lists of strings.

    Stratifies by STRATIFY_COL. If a country has fewer than 2 rows it
    cannot be stratified — those rows are added to the training set with
    a warning, since they're too small to guarantee test representation.
    """
    from sklearn.model_selection import train_test_split

    ids = df[CASE_ID_COL].astype(str).tolist()
    strata = df[STRATIFY_COL].astype(str).tolist()

    # Identify countries with only 1 row — can't stratify these
    from collections import Counter
    counts = Counter(strata)
    singleton_mask = [counts[s] < 2 for s in strata]

    if any(singleton_mask):
        singleton_ids = [i for i, m in zip(ids, singleton_mask) if m]
        singleton_countries = set(s for s, m in zip(strata, singleton_mask) if m)
        warnings.warn(
            f"The following countries have only 1 row and cannot be stratified. "
            f"Their rows are added to the training set: {sorted(singleton_countries)}. "
            f"Affected case_ids: {singleton_ids}"
        )
        # Remove singletons from the stratified split pool
        ids_to_split   = [i for i, m in zip(ids, singleton_mask) if not m]
        strata_to_split = [s for s, m in zip(strata, singleton_mask) if not m]
        extra_train_ids = singleton_ids
    else:
        ids_to_split    = ids
        strata_to_split = strata
        extra_train_ids = []

    if not ids_to_split:
        # Edge case: entire dataset is singletons
        return extra_train_ids, []

    train_ids, test_ids = train_test_split(
        ids_to_split,
        test_size=SPLIT_TEST_SIZE,   # 20% test
        random_state=SPLIT_SEED,     # fixed seed — never change
        stratify=strata_to_split,
    )

    return list(train_ids) + extra_train_ids, list(test_ids)