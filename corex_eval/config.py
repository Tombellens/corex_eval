"""
config.py
=========
Central configuration for the corex_eval library.

All column names, split parameters, file paths, and variable-to-column
mappings live here. If the dataset schema changes, this is the only file
that needs to be updated.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Contributors set COREX_DATA_DIR to wherever they store the data locally.
# The data folder is gitignored and never pushed to the repo.
# Example: export COREX_DATA_DIR=/home/tom/corex_data
_DATA_DIR = Path(os.environ.get("COREX_DATA_DIR", "data"))

GOLD_PATH       = _DATA_DIR / "gold"   / "corex_gold.csv"
SILVER_PATH     = _DATA_DIR / "silver" / "corex_silver.csv"
SILVER_EDU_PATH = _DATA_DIR / "silver" / "corex_silver_edu.csv"

RESULTS_PATH = Path("results") / "register.csv"


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

SPLIT_SEED      = 42      # never change — guarantees everyone uses the same test set
SPLIT_TEST_SIZE = 0.20
STRATIFY_COL    = "country_label"   # stratify split by country


# ---------------------------------------------------------------------------
# Core identifiers
# ---------------------------------------------------------------------------

CASE_ID_COL     = "case_id"
COUNTRY_COL     = "country_label"
MISSING_COL     = "missing_type"    # rows with a value here are incomplete records


# ---------------------------------------------------------------------------
# Collection task
# ---------------------------------------------------------------------------

COLLECTION_INPUT_COLS = [
    "case_id",
    "name_first",
    "name_last",
    "job_title",
    "country_label",
]

COLLECTION_TARGET_COL = "cv_links"   # pipe-separated gold URLs


# ---------------------------------------------------------------------------
# Extraction task
# ---------------------------------------------------------------------------

EXTRACTION_INPUT_COLS = [
    "case_id",
    "cv_local",
]

EXTRACTION_TEXT_COL = "cv_local"

# Atomic variables: single-value fields
# Maps variable name → (column, type)
# type is "str" | "int" — determines which metric to apply
ATOMIC_VARIABLES: dict[str, tuple[str, str]] = {
    "birth_year":    ("birth_year",  "int"),
    "birth_place":   ("birth_place", "str"),
    "birth_country": ("birth_country", "str"),  # note: stored as "17 = Belgium" etc.
    "sex":           ("sex",         "str"),     # stored as "0 = male" etc.
    "edu_start":     ("edu_start",   "int"),
    "edu_end":       ("edu_end",     "int"),
    "edu_degree":    ("edu_degree",  "str"),     # stored as "7 = Master's or equivalent"
}

# Composite variables: multi-record fields stored wide in the dataset
# Maps variable name → (start_year_pattern, end_year_pattern, position_pattern, max_spells)
COMPOSITE_VARIABLES: dict[str, tuple[str, str, str, int]] = {
    "career": (
        "career_start_year_{n}",
        "career_end_year_{n}",
        "career_position_{n}",
        20,
    ),
    "education": (
        "edu_start",   # education has only one entry in this schema
        "edu_end",
        "edu_degree",
        1,
    ),
}

# All valid extraction variable names (for validation in evaluate())
EXTRACTION_VARIABLES = set(ATOMIC_VARIABLES) | set(COMPOSITE_VARIABLES)


# ---------------------------------------------------------------------------
# Annotation task
# ---------------------------------------------------------------------------

# Maps annotation variable → gold column in the gold standard dataset
# and the expected silver column name in the silver standard
ANNOTATION_VARIABLES: dict[str, dict[str, str]] = {
    "career_position": {
        "gold_col":          "career_position",        # stored directly in silver rows (gold code)
        "silver_input_col":  "job_description_label",  # from silver standard
        "spell_index_col":   "spell_index",
    },
    "uni_subject": {
        "gold_col":          "uni_subject",      # stored directly in silver rows (gold code)
        "silver_input_col":  "subject_label",    # from silver standard
        "spell_index_col":   "spell_index",      # gold N from uni_subject_N; (case_id, spell_index) is the unique key
    },
}

# Secondary alignment key for spell-level annotation variables
SPELL_INDEX_COL = "spell_index"


# ---------------------------------------------------------------------------
# Temporal matching tolerance
# ---------------------------------------------------------------------------

# Maximum year difference still counted as a match for career spell alignment
TEMPORAL_TOLERANCE_YEARS = 1


# ---------------------------------------------------------------------------
# Submit (GitHub)
# ---------------------------------------------------------------------------

# Contributors set GITHUB_TOKEN in their environment for submit=True to work
GITHUB_TOKEN_ENV_VAR = "GITHUB_TOKEN"

# These should match your actual GitHub repo once it exists
GITHUB_REPO_OWNER = "your-github-username"   # update before first use
GITHUB_REPO_NAME  = "corex_eval"
GITHUB_RESULTS_FILE_PATH = "results/register.csv"