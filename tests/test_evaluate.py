"""
test_evaluate.py
================
End-to-end integration tests for corex_eval.evaluate().

Uses synthetic perfect/wrong predictions built from the gold data
to verify the full pipeline — alignment, metric computation, and
result structure — for each task and variable type.

Location in project:
    corex_eval/
    └── tests/
        └── test_evaluate.py
"""

import warnings
import pytest
import pandas as pd


def _eval(*args, **kwargs):
    """Wrapper that suppresses expected warnings during evaluate()."""
    from corex_eval import evaluate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate(*args, **kwargs)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class TestEvaluateCollection:

    def test_perfect_preds_give_f1_one(self, perfect_collection_preds):
        r = _eval(perfect_collection_preds, task="collection")
        assert r["macro"]["f1"] == pytest.approx(1.0)

    def test_empty_preds_give_recall_zero(self, empty_collection_preds):
        r = _eval(empty_collection_preds, task="collection")
        assert r["macro"]["recall"] == pytest.approx(0.0)

    def test_result_has_required_keys(self, perfect_collection_preds):
        r = _eval(perfect_collection_preds, task="collection")
        for key in ["macro", "per_case", "n_evaluated", "n_skipped", "task"]:
            assert key in r, f"Missing key: {key}"

    def test_macro_has_precision_recall_f1(self, perfect_collection_preds):
        r = _eval(perfect_collection_preds, task="collection")
        for key in ["precision", "recall", "f1"]:
            assert key in r["macro"]

    def test_task_is_set_correctly(self, perfect_collection_preds):
        r = _eval(perfect_collection_preds, task="collection")
        assert r["task"]     == "collection"
        assert r["variable"] is None

    def test_pipe_separated_urls_accepted(self, test_df):
        """retrieved_urls as pipe-separated string should give same result as list."""
        preds = pd.DataFrame([
            {
                "case_id":        str(row["case_id"]),
                "retrieved_urls": str(row.get("cv_links", "")),
            }
            for _, row in test_df.iterrows()
        ])
        r = _eval(preds, task="collection")
        assert r["macro"]["f1"] == pytest.approx(1.0)

    def test_n_evaluated_matches_cases_with_gold(self, perfect_collection_preds):
        r = _eval(perfect_collection_preds, task="collection")
        assert r["n_evaluated"] > 0


# ---------------------------------------------------------------------------
# Extraction — atomic integer (birth_year)
# ---------------------------------------------------------------------------

class TestEvaluateExtractionBirthYear:

    def test_perfect_gives_mae_zero(self, perfect_birth_year_preds):
        r = _eval(perfect_birth_year_preds, task="extraction", variable="birth_year")
        assert r["mae"]      == pytest.approx(0.0)
        assert r["accuracy"] == pytest.approx(1.0)

    def test_off_by_five_gives_mae_five(self, wrong_birth_year_preds):
        r = _eval(wrong_birth_year_preds, task="extraction", variable="birth_year")
        assert r["mae"]      == pytest.approx(5.0)
        assert r["accuracy"] == pytest.approx(0.0)

    def test_result_has_required_keys(self, perfect_birth_year_preds):
        r = _eval(perfect_birth_year_preds, task="extraction", variable="birth_year")
        for key in ["mae", "accuracy", "n_evaluated", "n_skipped", "task", "variable"]:
            assert key in r, f"Missing key: {key}"

    def test_task_and_variable_set_correctly(self, perfect_birth_year_preds):
        r = _eval(perfect_birth_year_preds, task="extraction", variable="birth_year")
        assert r["task"]     == "extraction"
        assert r["variable"] == "birth_year"

    def test_n_evaluated_is_positive(self, perfect_birth_year_preds):
        r = _eval(perfect_birth_year_preds, task="extraction", variable="birth_year")
        assert r["n_evaluated"] > 0


# ---------------------------------------------------------------------------
# Extraction — atomic string (birth_place)
# ---------------------------------------------------------------------------

class TestEvaluateExtractionBirthPlace:

    def test_perfect_gives_accuracy_one(self, test_df):
        preds = pd.DataFrame([
            {"case_id": str(row["case_id"]), "birth_place": str(row.get("birth_place", ""))}
            for _, row in test_df.iterrows()
            if str(row.get("birth_place", "")).strip() not in ("", "nan", "99", "/")
        ])
        r = _eval(preds, task="extraction", variable="birth_place")
        assert r["accuracy"] == pytest.approx(1.0)

    def test_returns_accuracy_not_mae(self, test_df):
        preds = pd.DataFrame([
            {"case_id": str(row["case_id"]), "birth_place": str(row.get("birth_place", ""))}
            for _, row in test_df.iterrows()
        ])
        r = _eval(preds, task="extraction", variable="birth_place")
        assert "accuracy" in r
        assert "mae" not in r


# ---------------------------------------------------------------------------
# Extraction — composite (career)
# ---------------------------------------------------------------------------

class TestEvaluateExtractionCareer:

    def test_perfect_gives_f1_one(self, perfect_career_preds):
        r = _eval(perfect_career_preds, task="extraction", variable="career")
        assert r["macro"]["f1"]        == pytest.approx(1.0)
        assert r["macro"]["precision"] == pytest.approx(1.0)
        assert r["macro"]["recall"]    == pytest.approx(1.0)

    def test_empty_preds_give_recall_zero(self, test_df):
        preds = pd.DataFrame([
            {"case_id": str(row["case_id"]), "career": []}
            for _, row in test_df.iterrows()
        ])
        r = _eval(preds, task="extraction", variable="career")
        assert r["macro"]["recall"] == pytest.approx(0.0)

    def test_result_has_required_keys(self, perfect_career_preds):
        r = _eval(perfect_career_preds, task="extraction", variable="career")
        for key in ["macro", "per_case", "n_evaluated",
                    "n_skipped", "tolerance_years", "task", "variable"]:
            assert key in r, f"Missing key: {key}"

    def test_per_case_has_detail(self, perfect_career_preds):
        r = _eval(perfect_career_preds, task="extraction", variable="career")
        first_case = next(iter(r["per_case"].values()))
        for key in ["precision", "recall", "f1", "n_predicted", "n_gold", "n_matched"]:
            assert key in first_case, f"Missing key '{key}' in per_case entry"

    def test_tolerance_years_recorded_in_result(self, perfect_career_preds):
        r = _eval(perfect_career_preds, task="extraction", variable="career",
                  tolerance_years=2)
        assert r["tolerance_years"] == 2

    def test_off_by_one_matches_with_default_tolerance(self, career_spells_test):
        """start_year off by 1 should still score perfectly with default tolerance=1."""
        preds_by_case = {}
        for _, row in career_spells_test.iterrows():
            case_id = str(row["case_id"])
            start   = int(row["career_start_year"]) + 1 if pd.notna(row["career_start_year"]) else None
            preds_by_case.setdefault(case_id, []).append({
                "start_year": start,
                "end_year":   int(row["career_end_year"]) if pd.notna(row["career_end_year"]) else None,
                "position":   str(row["career_position"]),
            })
        preds = pd.DataFrame([{"case_id": k, "career": v} for k, v in preds_by_case.items()])
        r = _eval(preds, task="extraction", variable="career", tolerance_years=1)
        assert r["macro"]["f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class TestEvaluateAnnotation:

    def test_perfect_gives_accuracy_one(self, perfect_annotation_preds):
        r = _eval(perfect_annotation_preds, task="annotation", variable="career_position")
        assert r["accuracy"]  == pytest.approx(1.0)
        assert r["macro_f1"]  == pytest.approx(1.0)

    def test_all_wrong_gives_accuracy_zero(self, wrong_annotation_preds):
        r = _eval(wrong_annotation_preds, task="annotation", variable="career_position")
        assert r["accuracy"] == pytest.approx(0.0)

    def test_result_has_required_keys(self, perfect_annotation_preds):
        r = _eval(perfect_annotation_preds, task="annotation", variable="career_position")
        for key in ["accuracy", "macro_f1", "weighted_f1", "per_class",
                    "semantic_similarity", "n_evaluated", "n_skipped", "task", "variable"]:
            assert key in r, f"Missing key: {key}"

    def test_semantic_similarity_none_by_default(self, perfect_annotation_preds):
        r = _eval(perfect_annotation_preds, task="annotation", variable="career_position")
        assert r["semantic_similarity"] is None

    def test_per_class_populated(self, perfect_annotation_preds):
        r = _eval(perfect_annotation_preds, task="annotation", variable="career_position")
        assert len(r["per_class"]) > 0

    def test_n_evaluated_is_positive(self, perfect_annotation_preds):
        r = _eval(perfect_annotation_preds, task="annotation", variable="career_position")
        assert r["n_evaluated"] > 0

    def test_task_and_variable_set_correctly(self, perfect_annotation_preds):
        r = _eval(perfect_annotation_preds, task="annotation", variable="career_position")
        assert r["task"]     == "annotation"
        assert r["variable"] == "career_position"


# ---------------------------------------------------------------------------
# Validation — all tasks
# ---------------------------------------------------------------------------

class TestEvaluateValidation:

    def test_rejects_unknown_task(self, perfect_birth_year_preds):
        with pytest.raises(ValueError, match="unknown_task"):
            _eval(perfect_birth_year_preds, task="unknown_task", variable="birth_year")

    def test_rejects_missing_variable_for_extraction(self, perfect_birth_year_preds):
        with pytest.raises(ValueError, match="variable"):
            _eval(perfect_birth_year_preds, task="extraction")

    def test_rejects_missing_variable_for_annotation(self, perfect_annotation_preds):
        with pytest.raises(ValueError, match="variable"):
            _eval(perfect_annotation_preds, task="annotation")

    def test_rejects_unknown_extraction_variable(self, perfect_birth_year_preds):
        with pytest.raises(ValueError):
            _eval(perfect_birth_year_preds, task="extraction", variable="not_a_column")

    def test_rejects_missing_case_id_column(self):
        bad_df = pd.DataFrame([{"birth_year": 1975}])
        with pytest.raises(ValueError, match="case_id"):
            _eval(bad_df, task="extraction", variable="birth_year")

    def test_rejects_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["case_id", "birth_year"])
        with pytest.raises(ValueError, match="empty"):
            _eval(empty_df, task="extraction", variable="birth_year")

    def test_rejects_submit_without_experiment_path(self, perfect_birth_year_preds):
        with pytest.raises(ValueError, match="experiment_path"):
            _eval(
                perfect_birth_year_preds,
                task="extraction",
                variable="birth_year",
                submit=True,
            )

    def test_warns_when_predictions_missing_for_some_test_cases(self, test_df):
        """Providing predictions for only half the test set should warn."""
        from corex_eval import evaluate
        half = test_df.head(max(1, len(test_df) // 2))
        preds = pd.DataFrame([
            {"case_id": str(row["case_id"]), "birth_year": row["birth_year"]}
            for _, row in half.iterrows()
            if pd.notna(row["birth_year"])
        ])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            evaluate(preds, task="extraction", variable="birth_year")
        warning_messages = [str(warning.message) for warning in w]
        assert any("no prediction" in msg.lower() for msg in warning_messages), (
            f"Expected a 'no prediction' warning. Got: {warning_messages}"
        )
