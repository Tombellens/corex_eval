"""
test_silver.py
==============
Unit tests for corex_eval.silver — load_silver() and get_silver_inputs().

These tests use synthetic DataFrames written to tmp_path. No COREX_DATA_DIR
is required — the session-scoped autouse fixture is overridden below.
"""

import warnings

import pandas as pd
import pytest


# Override the session-scoped autouse fixture so these tests are NOT
# skipped when COREX_DATA_DIR is missing.
@pytest.fixture(scope="session", autouse=True)
def require_data_dir():
    pass  # silver unit tests need no data files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_silver_csv(tmp_path, rows, filename="corex_silver.csv"):
    df = pd.DataFrame(rows)
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def _career_row(case_id, spell_index, job_desc="Engineer", workplace="Org"):
    return {
        "case_id":               case_id,
        "spell_index":           spell_index,
        "job_description_label": job_desc,
        "workplace_label":       workplace,
        "subject_label":         "",
        "uni_subject":           "",
    }


def _edu_subject_row(case_id, uni_subject, subject_label="", spell_index=1):
    """Education subject row — identified by (case_id, spell_index), gold code in uni_subject."""
    return {
        "case_id":               case_id,
        "spell_index":           spell_index,  # gold N from uni_subject_N
        "job_description_label": "",
        "workplace_label":       "",
        "subject_label":         subject_label,
        "uni_subject":           uni_subject,
    }


def _make_mixed_silver_df():
    """Minimal silver DataFrame with both career and edu rows."""
    rows = [
        {"case_id": "c1", "spell_index": 1, "job_description_label": "Engineer",
         "workplace_label": "OrgA", "subject_label": "", "uni_subject": ""},
        {"case_id": "c1", "spell_index": 2, "job_description_label": "Manager",
         "workplace_label": "OrgB", "subject_label": "", "uni_subject": ""},
        {"case_id": "c2", "spell_index": 1, "job_description_label": "Doctor",
         "workplace_label": "Hospital", "subject_label": "", "uni_subject": ""},
        # edu subject row — identified by (case_id, spell_index=1), gold code in uni_subject
        {"case_id": "c3", "spell_index": 1, "job_description_label": "",
         "workplace_label": "", "subject_label": "Political Science",
         "uni_subject": "601 = Political Science"},
    ]
    df = pd.DataFrame(rows)
    df["spell_index"] = pd.to_numeric(df["spell_index"], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# load_silver() — loading and basic cleaning
# ---------------------------------------------------------------------------

class TestLoadSilver:

    def test_loads_csv_and_returns_dataframe(self, tmp_path):
        from corex_eval.silver import load_silver
        path = _make_silver_csv(tmp_path, [_career_row("c1", 1)])
        df = load_silver(path)
        assert isinstance(df, pd.DataFrame)

    def test_drops_rows_with_empty_case_id(self, tmp_path):
        from corex_eval.silver import load_silver
        rows = [_career_row("c1", 1), _career_row("", 2)]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_silver(path)
        assert "" not in df["case_id"].tolist()
        assert len(df) == 1

    def test_drops_rows_with_whitespace_only_case_id(self, tmp_path):
        from corex_eval.silver import load_silver
        rows = [_career_row("c1", 1), _career_row("   ", 2)]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_silver(path)
        assert len(df) == 1

    def test_coerces_spell_index_to_int64(self, tmp_path):
        from corex_eval.silver import load_silver
        path = _make_silver_csv(tmp_path, [_career_row("c1", "3")])
        df = load_silver(path)
        assert str(df["spell_index"].dtype) == "Int64"

    def test_drops_rows_with_unparsable_spell_index(self, tmp_path):
        """Rows with non-empty content must have a valid spell_index."""
        from corex_eval.silver import load_silver
        rows = [_career_row("c1", 1), _career_row("c2", "not_a_number")]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_silver(path)
        assert len(df) == 1
        assert "c2" not in df["case_id"].tolist()

    def test_raises_file_not_found_when_path_missing(self, tmp_path):
        from corex_eval.silver import load_silver
        with pytest.raises(FileNotFoundError, match="corex_silver.csv"):
            load_silver(tmp_path / "nonexistent.csv")

    def test_drops_case_with_duplicate_career_spell_indices(self, tmp_path):
        from corex_eval.silver import load_silver
        rows = [
            _career_row("c_dup", 1, "Engineer", "OrgA"),
            _career_row("c_dup", 1, "Doctor",   "OrgB"),  # same spell_index
            _career_row("c_ok",  1),
        ]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_silver(path)
        assert "c_dup" not in df["case_id"].tolist()
        assert "c_ok" in df["case_id"].tolist()

    def test_warns_on_duplicate_spell_case(self, tmp_path):
        from corex_eval.silver import load_silver
        rows = [
            _career_row("c_dup", 1, "Engineer", "OrgA"),
            _career_row("c_dup", 1, "Doctor",   "OrgB"),
        ]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_silver(path)
        warning_texts = [str(warning.message) for warning in w]
        assert any("duplicate" in t.lower() or "c_dup" in t for t in warning_texts)

    def test_keeps_case_without_duplicates_intact(self, tmp_path):
        from corex_eval.silver import load_silver
        rows = [
            _career_row("c1", 1),
            _career_row("c1", 2),  # different spell_index — not a duplicate
            _career_row("c2", 1),
        ]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_silver(path)
        assert len(df) == 3

    def test_duplicate_check_is_per_variable(self, tmp_path):
        """
        A case with duplicate (case_id, spell_index) subject rows should be dropped,
        but a case with unique career spell_indices should be kept.
        """
        from corex_eval.silver import load_silver
        rows = [
            _edu_subject_row("c_dup", "601 = History", "Bachelor in History", spell_index=1),
            _edu_subject_row("c_dup", "601 = History", "Master in History",   spell_index=1),  # same index
            _career_row("c_ok", 1),
            _career_row("c_ok", 2),  # unique indices for c_ok
        ]
        path = _make_silver_csv(tmp_path, rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_silver(path)
        assert "c_dup" not in df["case_id"].tolist()
        assert "c_ok" in df["case_id"].tolist()


# ---------------------------------------------------------------------------
# load_silver() — edu file merging
# ---------------------------------------------------------------------------

class TestLoadSilverEduColumns:

    def test_edu_silver_file_has_correct_columns(self, tmp_path):
        from corex_eval.silver import load_silver
        rows = [_edu_subject_row("c1", uni_subject="601 = History",
                                 subject_label="Master in History")]
        path = _make_silver_csv(tmp_path, rows, "corex_silver_edu.csv")
        df = load_silver(path)
        assert "subject_label" in df.columns
        assert "uni_subject"   in df.columns
        assert "spell_index"   in df.columns

    def test_explicit_path_does_not_auto_merge_edu_file(self, tmp_path):
        """
        Auto-merge of corex_silver_edu.csv is only triggered when
        path=None (i.e. using the default SILVER_PATH from config).
        When an explicit path is given, no merge happens.
        """
        from corex_eval.silver import load_silver
        # Write career-only silver file
        rows = [_career_row("c1", 1)]
        career_path = _make_silver_csv(tmp_path, rows, "corex_silver.csv")
        # Also write an edu file in the same directory
        edu_rows = [_edu_subject_row("c2", "601 = History", "Master in History")]
        _make_silver_csv(tmp_path, edu_rows, "corex_silver_edu.csv")
        # Load via explicit career path — should NOT pick up edu rows
        df = load_silver(career_path)
        assert "c2" not in df["case_id"].tolist()


# ---------------------------------------------------------------------------
# get_silver_inputs() — variable routing and filtering
# ---------------------------------------------------------------------------

class TestGetSilverInputs:

    # --- career_position ---

    def test_career_position_returns_correct_columns(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        result = get_silver_inputs(df, "career_position", gold_case_ids={"c1", "c2", "c3"})
        assert "case_id"               in result.columns
        assert "spell_index"           in result.columns
        assert "job_description_label" in result.columns

    def test_career_position_excludes_rows_with_empty_label(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        result = get_silver_inputs(df, "career_position", gold_case_ids={"c1", "c2", "c3"})
        # c3 has empty job_description_label — should not appear
        assert "c3" not in result["case_id"].tolist()

    def test_career_position_includes_all_nonempty_career_rows(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        result = get_silver_inputs(df, "career_position", gold_case_ids={"c1", "c2", "c3"})
        # c1 has 2 spells, c2 has 1
        assert len(result) == 3

    # --- uni_subject ---

    def test_uni_subject_returns_correct_columns(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        result = get_silver_inputs(df, "uni_subject", gold_case_ids={"c1", "c2", "c3"})
        assert "case_id"       in result.columns
        assert "spell_index"   in result.columns
        assert "subject_label" in result.columns
        assert "uni_subject"   in result.columns

    def test_uni_subject_only_returns_rows_with_subject_label(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        result = get_silver_inputs(df, "uni_subject", gold_case_ids={"c1", "c2", "c3"})
        # Only c3 subject row has a non-empty subject_label
        assert list(result["case_id"]) == ["c3"]
        assert result.iloc[0]["subject_label"] == "Political Science"
        assert result.iloc[0]["uni_subject"] == "601 = Political Science"

    def test_uni_subject_returns_one_row_per_gold_slot(self):
        """Multiple subject rows for the same case (different spell_index) are all returned."""
        from corex_eval.silver import get_silver_inputs
        rows = [
            _edu_subject_row("c1", "02 = Arts", "Bachelor in History",   spell_index=1),
            _edu_subject_row("c1", "02 = Arts", "Master in Political History", spell_index=2),
        ]
        df = pd.DataFrame(rows)
        df["spell_index"] = pd.to_numeric(df["spell_index"], errors="coerce").astype("Int64")
        result = get_silver_inputs(df, "uni_subject", gold_case_ids={"c1"})
        assert len(result) == 2
        assert set(result["spell_index"].tolist()) == {1, 2}

    # --- gold alignment ---

    def test_filters_to_gold_case_ids(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        result = get_silver_inputs(df, "career_position", gold_case_ids={"c1"})
        assert all(result["case_id"] == "c1")

    def test_warns_when_silver_case_id_not_in_gold(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_silver_inputs(df, "career_position", gold_case_ids={"c1"})
        # c2 is in silver but not in gold_case_ids → should warn
        warning_texts = [str(warning.message) for warning in w]
        assert any("Dropped" in t or "not present" in t for t in warning_texts)

    def test_no_warning_when_all_case_ids_in_gold(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            get_silver_inputs(df, "career_position", gold_case_ids={"c1", "c2", "c3"})
        silver_warnings = [x for x in w if "silver" in str(x.message).lower()]
        assert len(silver_warnings) == 0

    # --- validation ---

    def test_raises_on_unknown_variable(self):
        from corex_eval.silver import get_silver_inputs
        df = _make_mixed_silver_df()
        with pytest.raises(ValueError, match="not a known annotation variable"):
            get_silver_inputs(df, "not_a_real_variable", gold_case_ids={"c1"})

    def test_raises_when_expected_column_missing_from_df(self):
        from corex_eval.silver import get_silver_inputs
        # DataFrame missing job_description_label entirely
        df = pd.DataFrame([
            {"case_id": "c1", "spell_index": pd.array([1], dtype="Int64")[0]}
        ])
        with pytest.raises(ValueError, match="job_description_label"):
            get_silver_inputs(df, "career_position", gold_case_ids={"c1"})
