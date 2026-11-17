"""
split.py
========
Deterministic train/dev/test split for the CoREx gold standard dataset.

This module is internal — contributors never call it directly. It is used
by inputs.py and evaluate.py to ensure everyone works with the exact same
test set.

The split is:
  - test: 20% of cases, stratified by country_label, seeded with SPLIT_SEED.
    This assignment never changes.
  - dev: DEV_SIZE of the remaining 80%, stratified by country, seeded with
    DEV_SEED. Used for prompt and design tuning.
  - train: the rest of the 80%.
  - Computed on the *cleaned* gold dataframe (incomplete records already
    removed before splitting).
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import TYPE_CHECKING

from corex_eval.config import (
    CASE_ID_COL,
    DEV_SEED,
    DEV_SIZE,
    SPLIT_SEED,
    SPLIT_TEST_SIZE,
    STRATIFY_COL,
)

if TYPE_CHECKING:
    import pandas as pd


def get_test_ids(df: "pd.DataFrame") -> set[str]:
    """Return the case_ids in the 20% test split."""
    _, test_ids = _split(df)
    return set(test_ids)


def get_train_ids(df: "pd.DataFrame", include_dev: bool = False) -> set[str]:
    """
    Return the case_ids in the training split.

    include_dev=False (default) excludes the dev set; include_dev=True
    returns the full 80% pool, for final runs after tuning is finished.
    """
    train_ids, _ = _split(df)
    if include_dev:
        return set(train_ids)
    return set(train_ids) - get_dev_ids(df)


def get_dev_ids(df: "pd.DataFrame") -> set[str]:
    """Return the case_ids in the dev split (a subset of the 80% train pool)."""
    train_ids, _ = _split(df)
    return set(_carve_dev(df, train_ids))


def apply_test_split(df: "pd.DataFrame") -> "pd.DataFrame":
    """Filter df to test rows only, index reset."""
    return _filter(df, get_test_ids(df))


def apply_dev_split(df: "pd.DataFrame") -> "pd.DataFrame":
    """Filter df to dev rows only, index reset."""
    return _filter(df, get_dev_ids(df))


def apply_train_split(df: "pd.DataFrame", include_dev: bool = False) -> "pd.DataFrame":
    """Filter df to training rows only, index reset. See get_train_ids()."""
    return _filter(df, get_train_ids(df, include_dev=include_dev))


def apply_eval_split(df: "pd.DataFrame", split: str) -> "pd.DataFrame":
    """Filter df to an evaluation split: "test" or "dev"."""
    if split == "test":
        return apply_test_split(df)
    if split == "dev":
        return apply_dev_split(df)
    raise ValueError(f"split must be 'test' or 'dev', got '{split}'.")


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _filter(df: "pd.DataFrame", ids: set[str]) -> "pd.DataFrame":
    return df[df[CASE_ID_COL].astype(str).isin(ids)].reset_index(drop=True)


def _split(df: "pd.DataFrame") -> tuple[list[str], list[str]]:
    """
    Core train/test split. Returns (train_ids, test_ids).

    Do not change: the test set assignment must stay identical across
    library versions. Countries with fewer than 2 rows cannot be stratified
    and go to the training set with a warning.
    """
    from sklearn.model_selection import train_test_split

    ids = df[CASE_ID_COL].astype(str).tolist()
    strata = df[STRATIFY_COL].astype(str).tolist()

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
        ids_to_split   = [i for i, m in zip(ids, singleton_mask) if not m]
        strata_to_split = [s for s, m in zip(strata, singleton_mask) if not m]
        extra_train_ids = singleton_ids
    else:
        ids_to_split    = ids
        strata_to_split = strata
        extra_train_ids = []

    if not ids_to_split:
        return extra_train_ids, []

    train_ids, test_ids = train_test_split(
        ids_to_split,
        test_size=SPLIT_TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=strata_to_split,
    )

    return list(train_ids) + extra_train_ids, list(test_ids)


def _carve_dev(df: "pd.DataFrame", train_ids: list[str]) -> list[str]:
    """
    Stratified dev sample from the train pool. Countries with fewer than
    2 rows in the pool stay in train.
    """
    from sklearn.model_selection import train_test_split

    country_of = dict(
        zip(df[CASE_ID_COL].astype(str), df[STRATIFY_COL].astype(str))
    )
    # Sorted so the result depends only on the set of train ids, not their order.
    pool = sorted(train_ids)
    strata = [country_of[i] for i in pool]
    counts = Counter(strata)
    keep = [counts[s] >= 2 for s in strata]
    pool   = [i for i, k in zip(pool, keep) if k]
    strata = [s for s, k in zip(strata, keep) if k]

    if len(pool) < 2:
        return []

    _, dev_ids = train_test_split(
        pool,
        test_size=DEV_SIZE,
        random_state=DEV_SEED,
        stratify=strata,
    )
    return list(dev_ids)
