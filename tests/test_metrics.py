"""
test_metrics.py
===============
Unit tests for corex_eval.metrics — collection, extraction, annotation.

These tests use synthetic data only. No COREX_DATA_DIR required.
They can run on any machine with pandas and scikit-learn installed.

Location in project:
    corex_eval/
    └── tests/
        └── test_metrics.py
"""

import pytest


# Override the session-scoped autouse fixture so these tests
# are NOT skipped when COREX_DATA_DIR is missing.
@pytest.fixture(scope="session", autouse=True)
def require_data_dir():
    pass   # metrics tests need no data files


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

class TestCollectionMetrics:

    def test_perfect_retrieval(self):
        from corex_eval.metrics.collection import collection_metrics
        r = collection_metrics(
            {"p1": {"url_a", "url_b"}, "p2": {"url_x"}},
            {"p1": {"url_a", "url_b"}, "p2": {"url_x"}},
        )
        assert r["macro"]["precision"] == pytest.approx(1.0)
        assert r["macro"]["recall"]    == pytest.approx(1.0)
        assert r["macro"]["f1"]        == pytest.approx(1.0)
        assert r["n_cases"] == 2

    def test_empty_retrieval(self):
        from corex_eval.metrics.collection import collection_metrics
        r = collection_metrics({"p1": set()}, {"p1": {"url_a", "url_b"}})
        assert r["macro"]["precision"] == pytest.approx(0.0)
        assert r["macro"]["recall"]    == pytest.approx(0.0)
        assert r["macro"]["f1"]        == pytest.approx(0.0)

    def test_partial_retrieval(self):
        from corex_eval.metrics.collection import collection_metrics
        # Retrieved 1 correct + 1 irrelevant; missed 1 gold
        r = collection_metrics({"p1": {"url_a", "url_z"}}, {"p1": {"url_a", "url_b"}})
        assert r["macro"]["precision"] == pytest.approx(0.5)
        assert r["macro"]["recall"]    == pytest.approx(0.5)
        assert r["per_case"]["p1"]["n_correct"] == 1

    def test_both_empty_counts_as_perfect(self):
        from corex_eval.metrics.collection import collection_metrics
        r = collection_metrics({"p1": set()}, {"p1": set()})
        assert r["macro"]["f1"] == pytest.approx(1.0)

    def test_macro_averages_across_cases(self):
        from corex_eval.metrics.collection import collection_metrics
        # p1 perfect (f1=1.0), p2 empty (f1=0.0) → macro f1 = 0.5
        r = collection_metrics(
            {"p1": {"url_a"}, "p2": set()},
            {"p1": {"url_a"}, "p2": {"url_x"}},
        )
        assert r["macro"]["f1"] == pytest.approx(0.5)

    def test_skips_prediction_with_no_gold(self):
        from corex_eval.metrics.collection import collection_metrics
        r = collection_metrics(
            {"p1": {"url_a"}, "p_unknown": {"url_b"}},
            {"p1": {"url_a"}},
        )
        assert r["n_skipped"] == 1
        assert r["n_cases"]   == 1

    def test_per_case_contains_expected_keys(self):
        from corex_eval.metrics.collection import collection_metrics
        r = collection_metrics({"p1": {"url_a"}}, {"p1": {"url_a"}})
        case = r["per_case"]["p1"]
        for key in ["precision", "recall", "f1", "n_retrieved", "n_gold", "n_correct"]:
            assert key in case, f"Missing key '{key}' in per_case"


# ---------------------------------------------------------------------------
# Extraction — atomic accuracy
# ---------------------------------------------------------------------------

class TestAtomicAccuracy:

    def test_perfect(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        r = atomic_accuracy(["Paris", "Berlin"], ["Paris", "Berlin"])
        assert r["accuracy"]  == pytest.approx(1.0)
        assert r["n_correct"] == 2
        assert r["n_total"]   == 2

    def test_all_wrong(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        r = atomic_accuracy(["London", "Rome"], ["Paris", "Berlin"])
        assert r["accuracy"] == pytest.approx(0.0)

    def test_case_insensitive_comparison(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        r = atomic_accuracy(["paris", "BERLIN"], ["Paris", "Berlin"])
        assert r["accuracy"] == pytest.approx(1.0)

    def test_partial_correct(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        r = atomic_accuracy(["Paris", "Rome"], ["Paris", "Berlin"])
        assert r["accuracy"] == pytest.approx(0.5)

    def test_null_gold_is_skipped(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        r = atomic_accuracy(["Paris", "Berlin"], [None, "Berlin"])
        assert r["n_skipped"] == 1
        assert r["n_total"]   == 1
        assert r["accuracy"]  == pytest.approx(1.0)

    def test_all_null_gold_returns_none(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        r = atomic_accuracy(["Paris"], [None])
        assert r["accuracy"] is None

    def test_sentinel_values_treated_as_empty(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        # "99" and "/" are sentinel values in the dataset meaning unknown
        r = atomic_accuracy(["Paris"], ["99"])
        assert r["n_skipped"] == 1

    def test_mismatched_lengths_raise(self):
        from corex_eval.metrics.extraction import atomic_accuracy
        with pytest.raises(ValueError):
            atomic_accuracy(["Paris", "Berlin"], ["Paris"])


# ---------------------------------------------------------------------------
# Extraction — atomic MAE
# ---------------------------------------------------------------------------

class TestAtomicMAE:

    def test_perfect(self):
        from corex_eval.metrics.extraction import atomic_mae
        r = atomic_mae([1975, 1980], [1975, 1980])
        assert r["mae"]      == pytest.approx(0.0)
        assert r["accuracy"] == pytest.approx(1.0)

    def test_constant_error(self):
        from corex_eval.metrics.extraction import atomic_mae
        r = atomic_mae([1980, 1985], [1975, 1980])
        assert r["mae"]      == pytest.approx(5.0)
        assert r["accuracy"] == pytest.approx(0.0)

    def test_mixed_errors(self):
        from corex_eval.metrics.extraction import atomic_mae
        # Errors: 0, 2, 4 → MAE = 2.0
        r = atomic_mae([1975, 1982, 1984], [1975, 1980, 1980])
        assert r["mae"] == pytest.approx(2.0)

    def test_null_prediction_is_skipped(self):
        from corex_eval.metrics.extraction import atomic_mae
        r = atomic_mae([None, 1980], [1975, 1980])
        assert r["n_skipped"] == 1
        assert r["n_total"]   == 1
        assert r["mae"]       == pytest.approx(0.0)

    def test_null_gold_is_skipped(self):
        from corex_eval.metrics.extraction import atomic_mae
        r = atomic_mae([1975, 1980], [None, 1980])
        assert r["n_skipped"] == 1

    def test_all_null_returns_none(self):
        from corex_eval.metrics.extraction import atomic_mae
        r = atomic_mae([None], [None])
        assert r["mae"] is None

    def test_mismatched_lengths_raise(self):
        from corex_eval.metrics.extraction import atomic_mae
        with pytest.raises(ValueError):
            atomic_mae([1975], [1975, 1980])


# ---------------------------------------------------------------------------
# Extraction — composite (career) metrics
# ---------------------------------------------------------------------------

class TestCompositeMetrics:

    def _spell(self, case_id, start, end, position="MP"):
        from corex_eval.metrics.extraction import CareerSpell
        return CareerSpell(case_id=case_id, start_year=start,
                           end_year=end, position=position)

    def test_perfect_match(self):
        from corex_eval.metrics.extraction import composite_metrics
        preds = {"p1": [self._spell("p1", 2010, 2015),
                        self._spell("p1", 2015, 2020)]}
        gold  = {"p1": [self._spell("p1", 2010, 2015),
                        self._spell("p1", 2015, 2020)]}
        r = composite_metrics(preds, gold)
        assert r["macro"]["precision"] == pytest.approx(1.0)
        assert r["macro"]["recall"]    == pytest.approx(1.0)
        assert r["macro"]["f1"]        == pytest.approx(1.0)

    def test_missing_spell_lowers_recall(self):
        from corex_eval.metrics.extraction import composite_metrics
        preds = {"p1": [self._spell("p1", 2010, 2015)]}
        gold  = {"p1": [self._spell("p1", 2010, 2015),
                        self._spell("p1", 2015, 2020)]}
        r = composite_metrics(preds, gold)
        assert r["macro"]["precision"] == pytest.approx(1.0)
        assert r["macro"]["recall"]    == pytest.approx(0.5)

    def test_extra_spell_lowers_precision(self):
        from corex_eval.metrics.extraction import composite_metrics
        preds = {"p1": [self._spell("p1", 2010, 2015),
                        self._spell("p1", 2016, 2020)]}
        gold  = {"p1": [self._spell("p1", 2010, 2015)]}
        r = composite_metrics(preds, gold)
        assert r["macro"]["precision"] == pytest.approx(0.5)
        assert r["macro"]["recall"]    == pytest.approx(1.0)

    def test_within_tolerance_matches(self):
        from corex_eval.metrics.extraction import composite_metrics
        # start_year off by 1 — should match with tolerance=1
        preds = {"p1": [self._spell("p1", 2011, 2015)]}
        gold  = {"p1": [self._spell("p1", 2010, 2015)]}
        r = composite_metrics(preds, gold, tolerance_years=1)
        assert r["macro"]["f1"] == pytest.approx(1.0)

    def test_outside_tolerance_does_not_match(self):
        from corex_eval.metrics.extraction import composite_metrics
        # start_year off by 2, tolerance=1 — should not match
        preds = {"p1": [self._spell("p1", 2012, 2015)]}
        gold  = {"p1": [self._spell("p1", 2010, 2015)]}
        r = composite_metrics(preds, gold, tolerance_years=1)
        assert r["macro"]["f1"] == pytest.approx(0.0)

    def test_none_years_are_always_compatible(self):
        from corex_eval.metrics.extraction import composite_metrics
        preds = {"p1": [self._spell("p1", 2010, None)]}
        gold  = {"p1": [self._spell("p1", 2010, None)]}
        r = composite_metrics(preds, gold)
        assert r["macro"]["f1"] == pytest.approx(1.0)

    def test_empty_prediction_gives_zero(self):
        from corex_eval.metrics.extraction import composite_metrics
        r = composite_metrics({"p1": []}, {"p1": [self._spell("p1", 2010, 2015)]})
        assert r["macro"]["f1"] == pytest.approx(0.0)

    def test_macro_averages_across_cases(self):
        from corex_eval.metrics.extraction import composite_metrics
        # p1 perfect (f1=1.0), p2 completely wrong (f1=0.0) → macro=0.5
        preds = {
            "p1": [self._spell("p1", 2010, 2015)],
            "p2": [self._spell("p2", 1999, 2001)],
        }
        gold = {
            "p1": [self._spell("p1", 2010, 2015)],
            "p2": [self._spell("p2", 2010, 2015)],
        }
        r = composite_metrics(preds, gold)
        assert r["macro"]["f1"] == pytest.approx(0.5)

    def test_skips_prediction_with_no_gold(self):
        from corex_eval.metrics.extraction import composite_metrics
        preds = {"p1": [self._spell("p1", 2010, 2015)], "p_unknown": []}
        gold  = {"p1": [self._spell("p1", 2010, 2015)]}
        r = composite_metrics(preds, gold)
        assert r["n_skipped"] == 1

    def test_spell_without_any_years_matches_nothing(self):
        from corex_eval.metrics.extraction import composite_metrics
        preds = {"p1": [self._spell("p1", None, None)]}
        gold  = {"p1": [self._spell("p1", 2010, 2015)]}
        r = composite_metrics(preds, gold)
        assert r["macro"]["f1"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

class TestAnnotationMetrics:

    def test_perfect(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["A", "B", "C"], ["A", "B", "C"])
        assert r["accuracy"]    == pytest.approx(1.0)
        assert r["macro_f1"]    == pytest.approx(1.0)
        assert r["weighted_f1"] == pytest.approx(1.0)

    def test_all_wrong(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["B", "C", "A"], ["A", "B", "C"])
        assert r["accuracy"] == pytest.approx(0.0)

    def test_partial_correct(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["A", "B", "A"], ["A", "B", "C"])
        assert r["accuracy"] == pytest.approx(2 / 3)

    def test_per_class_has_all_labels(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["A", "B", "A"], ["A", "B", "C"])
        for label in ["A", "B", "C"]:
            assert label in r["per_class"], f"Missing label '{label}' in per_class"

    def test_per_class_has_correct_fields(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["A"], ["A"])
        for field in ["precision", "recall", "f1", "support"]:
            assert field in r["per_class"]["A"], f"Missing field '{field}'"

    def test_n_total_and_n_classes(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["A", "B", "C"], ["A", "B", "C"])
        assert r["n_total"]   == 3
        assert r["n_classes"] == 3

    def test_empty_input_returns_none_metrics(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics([], [])
        assert r["accuracy"]  is None
        assert r["macro_f1"]  is None
        assert r["n_total"]   == 0

    def test_semantic_similarity_is_none_by_default(self):
        from corex_eval.metrics.annotation import annotation_metrics
        r = annotation_metrics(["A"], ["A"])
        assert r["semantic_similarity"] is None

    def test_mismatched_lengths_raise(self):
        from corex_eval.metrics.annotation import annotation_metrics
        with pytest.raises(ValueError):
            annotation_metrics(["A", "B"], ["A"])
