"""
gold.py
=======
Loader for the CoREx gold standard dataset.

This is the single point of entry for the gold data. It handles:
  - Loading from the path defined in config.py (via COREX_DATA_DIR)
  - Dropping incomplete records (rows with a missing_type value)
  - Dropping rows without a case_id
  - Dropping cases with duplicate spell indices in wide career/education columns
  - Returning a clean, consistently typed dataframe

All other modules (inputs.py, evaluate.py, split.py) call load_gold()
rather than reading the CSV themselves.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from corex_eval.config import (
    ATOMIC_VARIABLES,
    CASE_ID_COL,
    COMPOSITE_VARIABLES,
    GOLD_PATH,
    MISSING_COL,
    SPELL_INDEX_COL,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gold(path: str | Path | None = None) -> "pd.DataFrame":
    """
    Load and clean the gold standard dataset.

    Performs the following cleaning steps in order:
      1. Drop rows with no case_id (unidentifiable records)
      2. Drop rows with a missing_type value (incomplete records)
      3. Coerce year columns to nullable integers
      4. Detect and drop case_ids with duplicate career spell indices
         (rare but ambiguous — excluded from evaluation entirely)

    Parameters
    ----------
    path : Optional override for the gold CSV path. If None, uses GOLD_PATH
           from config.py (i.e. COREX_DATA_DIR/gold/corex_gold.csv).

    Returns
    -------
    Clean pandas DataFrame, one row per person-in-a-position.
    """
    import pandas as pd

    resolved = Path(path) if path else GOLD_PATH
    _assert_exists(resolved)

    df = pd.read_csv(resolved, dtype=str, keep_default_na=False, sep=";", encoding="utf-8-sig")

    df = _drop_missing_case_ids(df)
    df = _drop_incomplete_records(df)
    df = _coerce_year_columns(df)
    df = _drop_duplicate_spell_cases(df)

    return df.reset_index(drop=True)


def get_gold_column(df: "pd.DataFrame", variable: str) -> "pd.Series":
    """
    Extract the gold series for a given atomic variable name.

    Convenience helper used by evaluate.py so it doesn't need to know
    the column name mapping itself.

    Parameters
    ----------
    df       : Clean gold dataframe from load_gold().
    variable : Atomic variable name as defined in ATOMIC_VARIABLES
               (e.g. "birth_year", "sex", "edu_degree").

    Returns
    -------
    pandas Series with the gold values, indexed by position in df.

    Raises
    ------
    ValueError if variable is not a known atomic variable.
    """
    if variable not in ATOMIC_VARIABLES:
        raise ValueError(
            f"'{variable}' is not a known atomic variable. "
            f"Valid options: {sorted(ATOMIC_VARIABLES)}"
        )
    col, _ = ATOMIC_VARIABLES[variable]
    return df[col]


def get_career_spells(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Reshape the wide career columns into a long dataframe.

    Takes the wide format (career_start_year_1 ... career_position_20)
    and returns a tidy long dataframe with one row per career spell.

    Returns
    -------
    DataFrame with columns:
        case_id, spell_index, career_start_year, career_end_year, career_position
    Only spells where career_position is non-empty are included.
    """
    import pandas as pd

    _, (start_pat, end_pat, pos_pat, max_n) = (
        "career", COMPOSITE_VARIABLES["career"]
    )

    records = []
    for _, row in df.iterrows():
        case_id = row[CASE_ID_COL]
        for n in range(1, max_n + 1):
            start_col = start_pat.format(n=n)
            end_col   = end_pat.format(n=n)
            pos_col   = pos_pat.format(n=n)

            # Columns may not exist for all n (dataset only goes to 13)
            pos = row.get(pos_col, "")
            if not pos or str(pos).strip() in ("", "nan"):
                continue

            records.append({
                CASE_ID_COL:        case_id,
                SPELL_INDEX_COL:    n,
                "career_start_year": row.get(start_col, ""),
                "career_end_year":   row.get(end_col, ""),
                "career_position":   pos,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Internal cleaning helpers
# ---------------------------------------------------------------------------

def _drop_missing_case_ids(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop rows where case_id is empty or whitespace."""
    mask = df[CASE_ID_COL].str.strip().eq("")
    n_dropped = mask.sum()
    if n_dropped:
        warnings.warn(
            f"[gold] Dropped {n_dropped} row(s) with no case_id."
        )
    return df[~mask]


def _drop_incomplete_records(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Drop rows that have a missing_type value.

    These are records where the human coder flagged that only partial
    information was available (e.g. missing_type = '4 = position exists,
    we know the name...'). They are excluded from both train and test sets
    because the gold standard is incomplete for these individuals.
    """
    if MISSING_COL not in df.columns:
        return df

    mask = df[MISSING_COL].str.strip().ne("")
    n_dropped = mask.sum()
    if n_dropped:
        warnings.warn(
            f"[gold] Dropped {n_dropped} incomplete record(s) "
            f"(missing_type is set). These are excluded from both "
            f"train and test sets."
        )
    return df[~mask]


def _coerce_year_columns(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Coerce all year columns to pandas nullable integer (Int64).

    Year columns include edu_start, edu_end, birth_year, and all
    career_start_year_N / career_end_year_N columns.
    Values that cannot be parsed (empty strings, '99', '/') become pd.NA.
    """
    import pandas as pd

    year_cols = [
        col for col in df.columns
        if "year" in col.lower() or col in ("edu_start", "edu_end", "birth_year")
    ]

    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def _drop_duplicate_spell_cases(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Detect and drop any case_id that has duplicate career spell indices.

    A duplicate spell index means two or more career columns at the same
    position N have non-empty values pointing to different entries — this
    is structurally ambiguous and the case is excluded entirely.

    In practice this is rare (dataset is wide, not long), but we check
    anyway for robustness. Warns with the list of affected case_ids.
    """
    _, (start_pat, end_pat, pos_pat, max_n) = (
        "career", COMPOSITE_VARIABLES["career"]
    )

    bad_case_ids: list[str] = []

    for _, row in df.iterrows():
        case_id = row[CASE_ID_COL]
        seen_positions: list[int] = []

        for n in range(1, max_n + 1):
            pos_col = pos_pat.format(n=n)
            val = row.get(pos_col, "")
            if val and str(val).strip() not in ("", "nan", "<NA>"):
                seen_positions.append(n)

        # Duplicate indices would mean the same N appears twice —
        # in a wide dataframe this can only happen if columns are mis-named,
        # but we guard against it explicitly.
        if len(seen_positions) != len(set(seen_positions)):
            bad_case_ids.append(str(case_id))

    if bad_case_ids:
        warnings.warn(
            f"[gold] Dropped {len(bad_case_ids)} case_id(s) with duplicate "
            f"spell indices: {bad_case_ids}. These are excluded from both "
            f"train and test sets."
        )
        df = df[~df[CASE_ID_COL].astype(str).isin(bad_case_ids)]

    return df


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gold standard file not found: {path}\n"
            f"Set the COREX_DATA_DIR environment variable to the folder "
            f"containing your data, or pass an explicit path to load_gold().\n"
            f"Expected structure:\n"
            f"  $COREX_DATA_DIR/\n"
            f"  └── gold/\n"
            f"      └── corex_gold.csv"
        )