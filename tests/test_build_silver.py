"""
test_build_silver.py
====================
Unit tests for corex_eval.build_silver — pipeline helper functions.

These tests use only synthetic pandas Series / dicts. No COREX_DATA_DIR,
no OpenAI API key, and no file I/O are required.

Covers:
  - extract_spells_for_case()
  - extract_edu_for_case()
  - process_result()        (career GPT output parsing)
  - process_edu_result()    (education GPT output parsing)
  - build_user_prompt()     (career prompt construction)
  - build_edu_user_prompt() (education prompt construction)
"""

import pandas as pd
import pytest


# Override the session-scoped autouse fixture so these tests are NOT
# skipped when COREX_DATA_DIR is missing.
@pytest.fixture(scope="session", autouse=True)
def require_data_dir():
    pass  # build_silver unit tests need no data files


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row(**kwargs) -> pd.Series:
    """Build a pandas Series from kwargs (mimics a gold DataFrame row)."""
    return pd.Series(kwargs)


def _edu_anchor(degree="7 = Master", start=2000, end=2004, subjects=None):
    """Build a minimal education anchor dict."""
    if subjects is None:
        subjects = [{"subject_index": 1, "code": "Political Science", "uni_name": None}]
    return {"edu_degree": degree, "edu_start": start, "edu_end": end, "subjects": subjects}


# ---------------------------------------------------------------------------
# extract_spells_for_case
# ---------------------------------------------------------------------------

class TestExtractSpellsForCase:

    def test_returns_list_with_one_spell(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(career_position_1="MP", career_start_year_1="2000", career_end_year_1="2004")
        spells = extract_spells_for_case(row)
        assert len(spells) == 1
        assert spells[0]["spell_index"] == 1
        assert spells[0]["start_year"]  == 2000
        assert spells[0]["end_year"]    == 2004

    def test_skips_spell_when_position_is_empty(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(career_position_1="", career_start_year_1="2000", career_end_year_1="2004")
        assert extract_spells_for_case(row) == []

    def test_skips_spell_when_both_years_are_absent(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(career_position_1="MP", career_start_year_1="", career_end_year_1="")
        assert extract_spells_for_case(row) == []

    def test_keeps_spell_with_only_start_year(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(career_position_1="Mayor", career_start_year_1="1998", career_end_year_1="")
        spells = extract_spells_for_case(row)
        assert len(spells) == 1
        assert spells[0]["start_year"] == 1998
        assert spells[0]["end_year"]   is None

    def test_keeps_spell_with_only_end_year(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(career_position_1="Mayor", career_start_year_1="", career_end_year_1="2010")
        spells = extract_spells_for_case(row)
        assert len(spells) == 1
        assert spells[0]["start_year"] is None
        assert spells[0]["end_year"]   == 2010

    def test_extracts_multiple_spells(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(
            career_position_1="MP",    career_start_year_1="2000", career_end_year_1="2004",
            career_position_2="Mayor", career_start_year_2="2005", career_end_year_2="2009",
        )
        spells = extract_spells_for_case(row)
        assert len(spells) == 2
        assert spells[0]["spell_index"] == 1
        assert spells[1]["spell_index"] == 2

    def test_category_is_set_from_position_code(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(
            career_position_1="401 = politics, parliament",
            career_start_year_1="2000",
            career_end_year_1="2004",
        )
        spells = extract_spells_for_case(row)
        assert spells[0]["category"] == "401 = politics, parliament"

    def test_gap_in_spell_numbering_handled(self):
        """If spell 1 is empty but spell 2 is filled, spell 2 is still returned."""
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(
            career_position_1="",
            career_start_year_1="2000",
            career_end_year_1="2004",
            career_position_2="Mayor",
            career_start_year_2="2005",
            career_end_year_2="2009",
        )
        spells = extract_spells_for_case(row)
        assert len(spells) == 1
        assert spells[0]["spell_index"] == 2

    def test_nan_string_position_is_skipped(self):
        from corex_eval.build_silver import extract_spells_for_case
        row = _row(career_position_1="nan", career_start_year_1="2000", career_end_year_1="2004")
        assert extract_spells_for_case(row) == []


# ---------------------------------------------------------------------------
# extract_edu_for_case
# ---------------------------------------------------------------------------

class TestExtractEduForCase:

    def test_returns_none_when_no_subjects(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(edu_degree="7 = Master", edu_start="2000", edu_end="2002")
        assert extract_edu_for_case(row) is None

    def test_returns_anchor_with_one_subject(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(edu_degree="7 = Master", edu_start="2000", edu_end="2002",
                   uni_subject_1="Political Science")
        result = extract_edu_for_case(row)
        assert result is not None
        assert result["edu_degree"] == "7 = Master"
        assert result["edu_start"]  == 2000
        assert result["edu_end"]    == 2002
        assert len(result["subjects"]) == 1
        assert result["subjects"][0]["subject_index"] == 1
        assert result["subjects"][0]["code"] == "Political Science"

    def test_falls_back_to_unknown_when_edu_degree_missing(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Law")
        result = extract_edu_for_case(row)
        assert result["edu_degree"] == "unknown"

    def test_years_are_none_when_absent(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Economics")
        result = extract_edu_for_case(row)
        assert result["edu_start"] is None
        assert result["edu_end"]   is None

    def test_includes_uni_name_when_present(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Economics", uni_name_1="KU Leuven")
        result = extract_edu_for_case(row)
        assert result["subjects"][0]["uni_name"] == "KU Leuven"

    def test_uni_name_is_none_when_absent(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Economics")
        result = extract_edu_for_case(row)
        assert result["subjects"][0]["uni_name"] is None

    def test_extracts_multiple_subjects(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Economics", uni_subject_2="Political Science")
        result = extract_edu_for_case(row)
        assert len(result["subjects"]) == 2
        assert result["subjects"][0]["subject_index"] == 1
        assert result["subjects"][1]["subject_index"] == 2

    def test_skips_empty_string_subjects(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Economics", uni_subject_2="")
        result = extract_edu_for_case(row)
        assert len(result["subjects"]) == 1

    def test_skips_nan_string_subjects(self):
        from corex_eval.build_silver import extract_edu_for_case
        row = _row(uni_subject_1="Economics", uni_subject_2="nan")
        result = extract_edu_for_case(row)
        assert len(result["subjects"]) == 1


# ---------------------------------------------------------------------------
# process_result (career GPT output parsing)
# ---------------------------------------------------------------------------

class TestProcessResult:

    def _result(self, spells):
        return {"spells": spells}

    def _spell(self, idx, job="MP", workplace="Parliament", failed=False):
        return {
            "spell_index":           idx,
            "job_description_label": job,
            "workplace_label":       workplace,
            "extraction_failed":     failed,
        }

    def test_happy_path_all_spells_returned(self):
        from corex_eval.build_silver import process_result
        result = self._result([self._spell(1), self._spell(2, job="Mayor", workplace="City")])
        silver_rows, failed = process_result("c1", result, {1, 2})
        assert len(silver_rows) == 2
        assert failed == []

    def test_silver_row_has_correct_keys(self):
        from corex_eval.build_silver import process_result
        result = self._result([self._spell(1)])
        silver_rows, _ = process_result("c1", result, {1})
        row = silver_rows[0]
        for key in ["case_id", "spell_index", "job_description_label", "workplace_label"]:
            assert key in row, f"Missing key: {key}"

    def test_extraction_failed_flag_records_failure(self):
        from corex_eval.build_silver import process_result
        result = self._result([self._spell(1, failed=True)])
        silver_rows, failed = process_result("c1", result, {1})
        assert silver_rows == []
        assert len(failed) == 1
        assert failed[0]["reason"] == "extraction_failed_by_model"

    def test_null_labels_without_flag_also_fails(self):
        from corex_eval.build_silver import process_result
        result = self._result([{
            "spell_index": 1,
            "job_description_label": None,
            "workplace_label":       None,
            "extraction_failed":     False,
        }])
        silver_rows, failed = process_result("c1", result, {1})
        assert silver_rows == []
        assert len(failed) == 1

    def test_hallucinated_spell_index_is_ignored(self):
        from corex_eval.build_silver import process_result
        result = self._result([
            self._spell(1),
            self._spell(999, job="Ghost", workplace="Nowhere"),
        ])
        silver_rows, _ = process_result("c1", result, {1})
        assert len(silver_rows) == 1
        assert silver_rows[0]["spell_index"] == 1

    def test_missing_spell_recorded_as_failure(self):
        from corex_eval.build_silver import process_result
        # GPT only returns spell 1, but we expected {1, 2}
        result = self._result([self._spell(1)])
        _, failed = process_result("c1", result, {1, 2})
        missing_failures = [f for f in failed if f["spell_index"] == 2]
        assert len(missing_failures) == 1
        assert missing_failures[0]["reason"] == "spell_not_returned_by_model"

    def test_failure_includes_case_id(self):
        from corex_eval.build_silver import process_result
        result = self._result([self._spell(1, failed=True)])
        _, failed = process_result("case_42", result, {1})
        assert failed[0]["case_id"] == "case_42"

    def test_empty_spells_list_marks_all_as_missing(self):
        from corex_eval.build_silver import process_result
        result = self._result([])
        _, failed = process_result("c1", result, {1, 2, 3})
        assert len(failed) == 3
        assert all(f["reason"] == "spell_not_returned_by_model" for f in failed)


# ---------------------------------------------------------------------------
# process_edu_result (education GPT output parsing)
# ---------------------------------------------------------------------------

class TestProcessEduResult:

    def test_happy_path_degree_and_one_subject(self):
        from corex_eval.build_silver import process_edu_result
        result = {
            "degree_label": "Master in Political Sciences",
            "subjects": [
                {"subject_index": 1, "subject_label": "Political Sciences",
                 "extraction_failed": False},
            ],
        }
        silver_rows, failed = process_edu_result("c1", result, {1})
        assert len(silver_rows) == 2  # degree row (idx=0) + subject row (idx=1)
        assert failed == []

    def test_degree_row_uses_spell_index_zero(self):
        from corex_eval.build_silver import process_edu_result
        result = {"degree_label": "Master", "subjects": []}
        silver_rows, _ = process_edu_result("c1", result, set())
        degree_rows = [r for r in silver_rows if r["spell_index"] == 0]
        assert len(degree_rows) == 1
        assert degree_rows[0]["degree_label"] == "Master"

    def test_subject_rows_use_positive_spell_indices(self):
        from corex_eval.build_silver import process_edu_result
        result = {
            "degree_label": "PhD",
            "subjects": [
                {"subject_index": 2, "subject_label": "Law", "extraction_failed": False},
            ],
        }
        silver_rows, _ = process_edu_result("c1", result, {2})
        subject_rows = [r for r in silver_rows if r["spell_index"] == 2]
        assert len(subject_rows) == 1
        assert subject_rows[0]["subject_label"] == "Law"

    def test_null_degree_label_is_failure(self):
        from corex_eval.build_silver import process_edu_result
        result = {
            "degree_label": None,
            "subjects": [
                {"subject_index": 1, "subject_label": "Economics", "extraction_failed": False}
            ],
        }
        _, failed = process_edu_result("c1", result, {1})
        degree_failures = [f for f in failed if f["spell_index"] == 0]
        assert len(degree_failures) == 1
        assert degree_failures[0]["reason"] == "degree_label_not_found"

    def test_empty_string_degree_label_is_failure(self):
        from corex_eval.build_silver import process_edu_result
        result = {"degree_label": "", "subjects": []}
        _, failed = process_edu_result("c1", result, set())
        assert any(f["spell_index"] == 0 for f in failed)

    def test_subject_extraction_failed_flag_records_failure(self):
        from corex_eval.build_silver import process_edu_result
        result = {
            "degree_label": "Master",
            "subjects": [
                {"subject_index": 1, "subject_label": None, "extraction_failed": True},
            ],
        }
        _, failed = process_edu_result("c1", result, {1})
        subject_failures = [f for f in failed if f["spell_index"] == 1]
        assert len(subject_failures) == 1
        assert subject_failures[0]["reason"] == "extraction_failed_by_model"

    def test_hallucinated_subject_index_is_ignored(self):
        from corex_eval.build_silver import process_edu_result
        result = {
            "degree_label": "PhD",
            "subjects": [
                {"subject_index": 1,   "subject_label": "Law",   "extraction_failed": False},
                {"subject_index": 999, "subject_label": "Ghost", "extraction_failed": False},
            ],
        }
        silver_rows, _ = process_edu_result("c1", result, {1})
        subject_rows = [r for r in silver_rows if r["spell_index"] != 0]
        assert len(subject_rows) == 1
        assert subject_rows[0]["spell_index"] == 1

    def test_missing_subject_recorded_as_failure(self):
        from corex_eval.build_silver import process_edu_result
        # GPT returns nothing for subjects, but we expected {1, 2}
        result = {"degree_label": "PhD", "subjects": []}
        _, failed = process_edu_result("c1", result, {1, 2})
        subject_failures = [f for f in failed if f["spell_index"] in {1, 2}]
        assert len(subject_failures) == 2
        assert all(f["reason"] == "subject_not_returned_by_model" for f in subject_failures)

    def test_silver_row_has_correct_keys(self):
        from corex_eval.build_silver import process_edu_result
        result = {
            "degree_label": "Master",
            "subjects": [
                {"subject_index": 1, "subject_label": "Economics", "extraction_failed": False}
            ],
        }
        silver_rows, _ = process_edu_result("c1", result, {1})
        for row in silver_rows:
            for key in ["case_id", "spell_index", "degree_label", "subject_label"]:
                assert key in row, f"Missing key '{key}' in row {row}"

    def test_failure_includes_case_id(self):
        from corex_eval.build_silver import process_edu_result
        result = {"degree_label": None, "subjects": []}
        _, failed = process_edu_result("case_99", result, set())
        assert all(f["case_id"] == "case_99" for f in failed)


# ---------------------------------------------------------------------------
# build_user_prompt (career)
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:

    def _spell(self, idx=1, start=2000, end=2004, category="MP"):
        return {"spell_index": idx, "start_year": start, "end_year": end, "category": category}

    def test_contains_cv_text(self):
        from corex_eval.build_silver import build_user_prompt
        result = build_user_prompt("My long CV text.", [self._spell()])
        assert "My long CV text." in result

    def test_contains_spell_year_range(self):
        from corex_eval.build_silver import build_user_prompt
        result = build_user_prompt("CV.", [self._spell(start=1995, end=2001)])
        assert "1995" in result
        assert "2001" in result

    def test_uses_question_mark_for_none_start_year(self):
        from corex_eval.build_silver import build_user_prompt
        result = build_user_prompt("CV.", [self._spell(start=None, end=2010)])
        assert "?" in result

    def test_uses_question_mark_for_none_end_year(self):
        from corex_eval.build_silver import build_user_prompt
        result = build_user_prompt("CV.", [self._spell(start=2005, end=None)])
        assert "?" in result

    def test_contains_category_hint(self):
        from corex_eval.build_silver import build_user_prompt
        result = build_user_prompt("CV.", [self._spell(category="401 = politics, parliament")])
        assert "401 = politics, parliament" in result

    def test_contains_spell_index(self):
        from corex_eval.build_silver import build_user_prompt
        result = build_user_prompt("CV.", [self._spell(idx=3)])
        assert "3" in result

    def test_multiple_spells_all_present(self):
        from corex_eval.build_silver import build_user_prompt
        spells = [self._spell(1, 2000, 2004), self._spell(2, 2005, 2009)]
        result = build_user_prompt("CV.", spells)
        assert "2000" in result
        assert "2005" in result


# ---------------------------------------------------------------------------
# build_edu_user_prompt (education)
# ---------------------------------------------------------------------------

class TestBuildEduUserPrompt:

    def _anchor(self, degree="7 = Master", start=2000, end=2004,
                subjects=None):
        if subjects is None:
            subjects = [{"subject_index": 1, "code": "Political Science", "uni_name": None}]
        return {"edu_degree": degree, "edu_start": start, "edu_end": end,
                "subjects": subjects}

    def test_contains_cv_text(self):
        from corex_eval.build_silver import build_edu_user_prompt
        result = build_edu_user_prompt("My detailed CV.", self._anchor())
        assert "My detailed CV." in result

    def test_contains_degree_level(self):
        from corex_eval.build_silver import build_edu_user_prompt
        result = build_edu_user_prompt("CV.", self._anchor(degree="7 = Master"))
        assert "7 = Master" in result

    def test_contains_period_when_years_present(self):
        from corex_eval.build_silver import build_edu_user_prompt
        result = build_edu_user_prompt("CV.", self._anchor(start=1999, end=2003))
        assert "1999" in result
        assert "2003" in result

    def test_omits_period_line_when_no_years(self):
        from corex_eval.build_silver import build_edu_user_prompt
        result = build_edu_user_prompt("CV.", self._anchor(start=None, end=None))
        assert "Period:" not in result

    def test_contains_subject_index_and_code(self):
        from corex_eval.build_silver import build_edu_user_prompt
        subjects = [{"subject_index": 3, "code": "Economics", "uni_name": None}]
        result = build_edu_user_prompt("CV.", self._anchor(subjects=subjects))
        assert "subject_index 3" in result
        assert "Economics" in result

    def test_includes_university_name_when_present(self):
        from corex_eval.build_silver import build_edu_user_prompt
        subjects = [{"subject_index": 1, "code": "Law", "uni_name": "KU Leuven"}]
        result = build_edu_user_prompt("CV.", self._anchor(subjects=subjects))
        assert "KU Leuven" in result

    def test_omits_university_hint_when_name_is_none(self):
        from corex_eval.build_silver import build_edu_user_prompt
        subjects = [{"subject_index": 1, "code": "Law", "uni_name": None}]
        result = build_edu_user_prompt("CV.", self._anchor(subjects=subjects))
        assert "university:" not in result.lower() or "None" not in result

    def test_multiple_subjects_all_present(self):
        from corex_eval.build_silver import build_edu_user_prompt
        subjects = [
            {"subject_index": 1, "code": "Economics",         "uni_name": None},
            {"subject_index": 2, "code": "Political Science",  "uni_name": None},
        ]
        result = build_edu_user_prompt("CV.", self._anchor(subjects=subjects))
        assert "Economics"        in result
        assert "Political Science" in result
