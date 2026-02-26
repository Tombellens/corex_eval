"""
conftest.py
===========
Shared pytest fixtures for the corex_eval test suite.

Requires COREX_DATA_DIR to be set before running:
    export COREX_DATA_DIR=/path/to/your/data

Then run with:
    pytest tests/ -v

Location in project:
    corex_eval/
    └── tests/
        └── conftest.py
"""

import os
import warnings
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
# Session-level: skip everything if COREX_DATA_DIR is not set
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def require_data_dir():
    if not os.environ.get("COREX_DATA_DIR"):
        pytest.skip(
            "COREX_DATA_DIR not set — skipping all data-dependent tests.\n"
            "Set it with:  export COREX_DATA_DIR=/path/to/your/data"
        )


# ---------------------------------------------------------------------------
# Gold and split fixtures — loaded once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gold_df():
    """Full cleaned gold dataframe (before split)."""
    from corex_eval.gold import load_gold
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_gold()


@pytest.fixture(scope="session")
def test_df(gold_df):
    """20% test split."""
    from corex_eval.split import apply_test_split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return apply_test_split(gold_df)


@pytest.fixture(scope="session")
def train_df(gold_df):
    """80% training split."""
    from corex_eval.split import apply_train_split
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return apply_train_split(gold_df)


@pytest.fixture(scope="session")
def career_spells_test(test_df):
    """Career spells in long format, test set only."""
    from corex_eval.gold import get_career_spells
    return get_career_spells(test_df)


# ---------------------------------------------------------------------------
# Prediction fixtures — synthetic predictions built from gold
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def perfect_collection_preds(test_df):
    """Predictions that exactly match gold cv_links."""
    rows = []
    for _, row in test_df.iterrows():
        raw  = str(row.get("cv_links", "")).strip()
        urls = [u.strip() for u in raw.split("|") if u.strip()]
        rows.append({"case_id": str(row["case_id"]), "retrieved_urls": urls})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def empty_collection_preds(test_df):
    """Predictions with no URLs retrieved for any case."""
    return pd.DataFrame([
        {"case_id": str(row["case_id"]), "retrieved_urls": []}
        for _, row in test_df.iterrows()
    ])


@pytest.fixture(scope="session")
def perfect_birth_year_preds(test_df):
    """Predictions that exactly match gold birth_year."""
    return pd.DataFrame([
        {"case_id": str(row["case_id"]), "birth_year": row["birth_year"]}
        for _, row in test_df.iterrows()
        if pd.notna(row["birth_year"])
    ])


@pytest.fixture(scope="session")
def wrong_birth_year_preds(test_df):
    """Predictions that are always 5 years off."""
    return pd.DataFrame([
        {"case_id": str(row["case_id"]), "birth_year": int(row["birth_year"]) + 5}
        for _, row in test_df.iterrows()
        if pd.notna(row["birth_year"])
    ])


@pytest.fixture(scope="session")
def perfect_career_preds(career_spells_test):
    """Predictions that exactly match gold career spells."""
    preds_by_case = {}
    for _, row in career_spells_test.iterrows():
        case_id = str(row["case_id"])
        preds_by_case.setdefault(case_id, []).append({
            "start_year": int(row["career_start_year"]) if pd.notna(row["career_start_year"]) else None,
            "end_year":   int(row["career_end_year"])   if pd.notna(row["career_end_year"])   else None,
            "position":   str(row["career_position"]),
        })
    return pd.DataFrame([
        {"case_id": k, "career": v}
        for k, v in preds_by_case.items()
    ])


@pytest.fixture(scope="session")
def perfect_annotation_preds(career_spells_test):
    """Perfect annotation predictions: uses gold career_position codes."""
    return pd.DataFrame([
        {
            "case_id":        str(row["case_id"]),
            "spell_index":    int(row["spell_index"]),
            "predicted_code": str(row["career_position"]),
        }
        for _, row in career_spells_test.iterrows()
    ])


@pytest.fixture(scope="session")
def wrong_annotation_preds(career_spells_test):
    """Annotation predictions that are always wrong."""
    return pd.DataFrame([
        {
            "case_id":        str(row["case_id"]),
            "spell_index":    int(row["spell_index"]),
            "predicted_code": "000 = definitely_wrong_label",
        }
        for _, row in career_spells_test.iterrows()
    ])


@pytest.fixture(scope="session")
def perfect_edu_start_preds(test_df):
    """Predictions that exactly match gold edu_start."""
    return pd.DataFrame([
        {"case_id": str(row["case_id"]), "edu_start": row["edu_start"]}
        for _, row in test_df.iterrows()
        if pd.notna(row.get("edu_start"))
    ])


@pytest.fixture(scope="session")
def perfect_edu_end_preds(test_df):
    """Predictions that exactly match gold edu_end."""
    return pd.DataFrame([
        {"case_id": str(row["case_id"]), "edu_end": row["edu_end"]}
        for _, row in test_df.iterrows()
        if pd.notna(row.get("edu_end"))
    ])


@pytest.fixture(scope="session")
def perfect_edu_degree_preds(test_df):
    """Predictions that exactly match gold edu_degree (atomic string)."""
    return pd.DataFrame([
        {"case_id": str(row["case_id"]), "edu_degree": str(row.get("edu_degree", ""))}
        for _, row in test_df.iterrows()
        if str(row.get("edu_degree", "")).strip() not in ("", "nan")
    ])
