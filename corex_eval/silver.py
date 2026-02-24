"""
silver.py
=========
Loader for the CoREx silver standard dataset.

The silver standard is a separate file containing LLM-generated labels
for career spells and education entries. It is used as input for the
annotation task — contributors receive a raw extracted string (the silver
label) and must map it to the correct gold codebook category.

Unlike the gold standard (wide format, one row per person-in-a-position),
the silver standard is in LONG format:
  - One row per career spell  (for career_position and related variables)
  - One row per education entry (for edu_degree, uni_subject)

Expected columns in corex_silver.csv
--------------------------------------
For career spell rows:
    case_id, spell_index, job_description_label, workplace_label

For education entry rows:
    case_id, spell_index, degree_label, subject_label

Both row types live in the same file, distinguished by which label
columns are populated.

Cleaning steps (mirrors gold.py):
  1. Drop rows with no case_id
  2. Drop case_ids with duplicate spell indices (per variable type)
  3. Align to the gold test/train split (only retain case_ids present
     in the gold standard after cleaning)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from corex_eval.config import (
    ANNOTATION_VARIABLES,
    CASE_ID_COL,
    SILVER_PATH,
    SPELL_INDEX_COL,
)

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_silver(path: str | Path | None = None) -> "pd.DataFrame":
    """
    Load and clean the silver standard dataset.

    Performs the following cleaning steps:
      1. Drop rows with no case_id
      2. Drop case_ids with duplicate spell indices (separately for
         career and education variable groups)
      3. Coerce spell_index to integer

    Parameters
    ----------
    path : Optional override for the silver CSV path. If None, uses
           SILVER_PATH from config.py (COREX_DATA_DIR/silver/corex_silver.csv).

    Returns
    -------
    Clean pandas DataFrame in long format.
    """
    import pandas as pd

    resolved = Path(path) if path else SILVER_PATH
    _assert_exists(resolved)

    df = pd.read_csv(resolved, dtype=str, keep_default_na=False)

    df = _drop_missing_case_ids(df)
    df = _coerce_spell_index(df)
    df = _drop_duplicate_spell_cases(df)

    return df.reset_index(drop=True)


def get_silver_inputs(
    df: "pd.DataFrame",
    variable: str,
    gold_case_ids: "set[str]",
) -> "pd.DataFrame":
    """
    Extract the silver standard input rows for a given annotation variable,
    restricted to case_ids present in the (cleaned) gold standard.

    This is the function called by load_inputs(task="annotation", variable=...).

    Parameters
    ----------
    df            : Clean silver dataframe from load_silver().
    variable      : Annotation variable name as defined in ANNOTATION_VARIABLES
                    (e.g. "career_position", "edu_degree", "uni_subject").
    gold_case_ids : Set of case_ids from the cleaned gold standard (after
                    incomplete records removed). Used to align silver to gold.

    Returns
    -------
    DataFrame with columns: [case_id, spell_index, <input_col>]
    where <input_col> is the raw label column for the given variable
    (e.g. "job_description_label" for career_position).

    Only rows where the input column is non-empty and the case_id exists
    in the gold standard are returned.
    """
    _validate_variable(variable)

    var_config   = ANNOTATION_VARIABLES[variable]
    input_col    = var_config["silver_input_col"]
    has_spells   = var_config["spell_index_col"] is not None

    # Check the expected input column exists
    if input_col not in df.columns:
        raise ValueError(
            f"Expected column '{input_col}' not found in silver standard. "
            f"Available columns: {list(df.columns)}"
        )

    # Filter to rows where the input label is populated
    mask_filled = df[input_col].str.strip().ne("")
    df_var = df[mask_filled].copy()

    # Align to gold case_ids
    before = len(df_var)
    df_var = df_var[df_var[CASE_ID_COL].astype(str).isin(gold_case_ids)]
    n_dropped = before - len(df_var)
    if n_dropped:
        warnings.warn(
            f"[silver] Dropped {n_dropped} row(s) for variable '{variable}' "
            f"whose case_ids are not present in the gold standard."
        )

    # Select and return relevant columns
    cols = [CASE_ID_COL, input_col]
    if has_spells and SPELL_INDEX_COL in df_var.columns:
        cols = [CASE_ID_COL, SPELL_INDEX_COL, input_col]

    return df_var[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal cleaning helpers
# ---------------------------------------------------------------------------

def _drop_missing_case_ids(df: "pd.DataFrame") -> "pd.DataFrame":
    """Drop rows where case_id is empty or whitespace."""
    mask = df[CASE_ID_COL].str.strip().eq("")
    n_dropped = mask.sum()
    if n_dropped:
        warnings.warn(
            f"[silver] Dropped {n_dropped} row(s) with no case_id."
        )
    return df[~mask]


def _coerce_spell_index(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Coerce spell_index to nullable integer.
    Rows where spell_index cannot be parsed are dropped with a warning.
    """
    import pandas as pd

    if SPELL_INDEX_COL not in df.columns:
        return df

    original_len = len(df)
    df[SPELL_INDEX_COL] = pd.to_numeric(
        df[SPELL_INDEX_COL], errors="coerce"
    ).astype("Int64")

    bad_mask = df[SPELL_INDEX_COL].isna()
    n_bad = bad_mask.sum()
    if n_bad:
        warnings.warn(
            f"[silver] Dropped {n_bad} row(s) where spell_index could not "
            f"be parsed as an integer."
        )
        df = df[~bad_mask]

    return df


def _drop_duplicate_spell_cases(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Drop any case_id that has duplicate (case_id, spell_index) pairs.

    Checked separately for each annotation variable group (career spells
    and education entries) because a case_id may legitimately appear in
    both groups without being a duplicate.

    Mirrors the same logic applied to the gold standard in gold.py —
    if a case_id is dropped from the silver standard, contributors will
    never receive it as an input and cannot submit predictions for it.
    """
    if SPELL_INDEX_COL not in df.columns:
        return df

    # Group variables by their spell_index_col to determine which rows
    # belong to which evaluation unit
    spell_vars = [
        v for v, cfg in ANNOTATION_VARIABLES.items()
        if cfg["spell_index_col"] is not None
    ]

    bad_case_ids: set[str] = set()

    for variable in spell_vars:
        input_col = ANNOTATION_VARIABLES[variable]["silver_input_col"]
        if input_col not in df.columns:
            continue

        # Only look at rows relevant to this variable
        subset = df[df[input_col].str.strip().ne("")][[CASE_ID_COL, SPELL_INDEX_COL]]

        duplicated = subset.duplicated(subset=[CASE_ID_COL, SPELL_INDEX_COL], keep=False)
        dup_ids = set(subset[duplicated][CASE_ID_COL].astype(str).tolist())

        if dup_ids:
            bad_case_ids |= dup_ids
            warnings.warn(
                f"[silver] Variable '{variable}': found duplicate "
                f"(case_id, spell_index) pairs for {len(dup_ids)} case_id(s): "
                f"{sorted(dup_ids)}. These case_ids are excluded entirely."
            )

    if bad_case_ids:
        df = df[~df[CASE_ID_COL].astype(str).isin(bad_case_ids)]

    return df.reset_index(drop=True)


def _validate_variable(variable: str) -> None:
    if variable not in ANNOTATION_VARIABLES:
        raise ValueError(
            f"'{variable}' is not a known annotation variable. "
            f"Valid options: {sorted(ANNOTATION_VARIABLES)}"
        )


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Silver standard file not found: {path}\n"
            f"Set the COREX_DATA_DIR environment variable to the folder "
            f"containing your data, or pass an explicit path to load_silver().\n"
            f"Expected structure:\n"
            f"  $COREX_DATA_DIR/\n"
            f"  └── silver/\n"
            f"      └── corex_silver.csv"
        )