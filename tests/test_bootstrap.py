"""
Bootstrap CI tests — no data needed.
"""

import random

import pytest


@pytest.fixture(scope="session", autouse=True)
def require_data_dir():
    """Override the conftest skip: these tests need no data."""
    return


def _labels(n=400, seed=1):
    rng = random.Random(seed)
    classes = ["1", "2", "3", "4", "6", "7"]
    weights = [30, 25, 20, 15, 7, 3]
    gold = rng.choices(classes, weights, k=n)
    pred = [g if rng.random() < 0.7 else rng.choice(classes) for g in gold]
    cases = [f"case_{i // 3}" for i in range(n)]  # three spells per person
    return gold, pred, cases


def test_classification_stats_match_sklearn():
    from sklearn.metrics import accuracy_score, f1_score
    from corex_eval.metrics.bootstrap import classification_stats
    gold, pred, _ = _labels()
    s = classification_stats(gold, pred)(list(range(len(gold))))
    assert s["accuracy"]    == pytest.approx(accuracy_score(gold, pred))
    assert s["macro_f1"]    == pytest.approx(f1_score(gold, pred, average="macro", zero_division=0))
    assert s["weighted_f1"] == pytest.approx(f1_score(gold, pred, average="weighted", zero_division=0))


def test_classification_stats_label_set_follows_sample():
    from sklearn.metrics import f1_score
    from corex_eval.metrics.bootstrap import classification_stats
    gold = ["1", "1", "2", "9"]
    pred = ["1", "2", "2", "1"]
    idx = [0, 1, 2]  # label "9" absent from this sample
    s = classification_stats(gold, pred)(idx)
    sub_g, sub_p = [gold[i] for i in idx], [pred[i] for i in idx]
    assert s["macro_f1"] == pytest.approx(f1_score(sub_g, sub_p, average="macro", zero_division=0))


def test_ci_brackets_point_estimate_and_is_deterministic():
    from corex_eval.metrics.bootstrap import classification_stats, cluster_bootstrap_ci
    gold, pred, cases = _labels()
    stat = classification_stats(gold, pred)
    point = stat(list(range(len(gold))))
    ci = cluster_bootstrap_ci(cases, stat, n_boot=300)
    for name in ("accuracy", "macro_f1", "weighted_f1"):
        lo, hi = ci[name]
        assert lo <= point[name] <= hi
        assert hi - lo > 0
    assert ci == cluster_bootstrap_ci(cases, stat, n_boot=300)


def test_cluster_resampling_widens_interval_for_correlated_items():
    from corex_eval.metrics.bootstrap import cluster_bootstrap_ci, mean_of
    # Each person is either all-correct or all-wrong across 10 items.
    values = [float(p % 2) for p in range(60) for _ in range(10)]
    clusters_person = [f"p{p}" for p in range(60) for _ in range(10)]
    clusters_item = [str(i) for i in range(len(values))]
    def stat(idx):
        return {"m": mean_of(values)(idx)}

    def width(clusters):
        lo, hi = cluster_bootstrap_ci(clusters, stat, n_boot=300)["m"]
        return hi - lo

    assert width(clusters_person) > 2 * width(clusters_item)


def test_mean_of_ignores_none_and_handles_all_none():
    from corex_eval.metrics.bootstrap import mean_of
    assert mean_of([1.0, None, 3.0])([0, 1, 2]) == 2.0
    assert mean_of([None, None])([0, 1]) is None


def test_zero_replicates_disables():
    from corex_eval.metrics.bootstrap import cluster_bootstrap_ci
    assert cluster_bootstrap_ci(["a"], lambda idx: {"m": 1.0}, n_boot=0) == {}


def test_atomic_ci_uses_headline_definition():
    from corex_eval.evaluate import _atomic_ci
    ids = [f"c{i}" for i in range(50)]
    preds = ["Vienna"] * 25 + ["Graz"] * 25
    golds = ["Vienna"] * 50
    ci = _atomic_ci(ids, preds, golds, "birth_place", "str", 200)
    assert 0.0 < ci["accuracy"][0] <= ci["accuracy"][1] <= 1.0
    ci_int = _atomic_ci(ids, [1960] * 50, [1960] * 25 + [1961] * 25, "birth_year", "int", 200)
    assert ci_int["mae"][0] < 0.5 < ci_int["mae"][1]


class TestAnnotationOutputChecks:

    def _checks(self, preds, golds, granularity="fine", variable="career_position"):
        from corex_eval.evaluate import _annotation_output_checks
        known = {"105", "201", "401"}
        return _annotation_output_checks(preds, golds, known, granularity, variable)

    def test_invalid_counts_unknown_and_unparseable(self):
        c = self._checks(["105 = Minister", "999 = Other", "I am not sure", ""],
                         ["105 = Minister"] * 4)
        assert c["n_invalid"] == 3 and c["invalid_rate"] == 0.75

    def test_format_mismatch_counts_right_code_in_other_format(self):
        c = self._checks(["105", "105 = Minister", "201"], ["105 = Minister"] * 3)
        assert c["n_format_mismatch"] == 1
        assert c["n_invalid"] == 0

    def test_broad_validity_uses_first_digit(self):
        c = self._checks(["1", "4", "9"], ["1", "4", "1"], granularity="broad")
        assert c["n_invalid"] == 1
        assert "n_format_mismatch" not in c


class TestCountryKey:

    def _key(self, v, names=None):
        from corex_eval.evaluate import _country_key
        return _country_key(v, names or {"17": "Belgium"})

    def test_coded_and_free_text_agree(self):
        assert self._key("17 = Belgium") == self._key("Belgium") == self._key("belgium ")

    def test_bare_code_resolved_from_gold(self):
        assert self._key("17") == "belgium"

    def test_aliases_and_accents(self):
        assert self._key("UK") == self._key("Great Britain") == self._key("United Kingdom")
        assert self._key("Türkiye") == self._key("Turkey")
        assert self._key("The Netherlands") == self._key("Netherlands")
        assert self._key("Österreich") == self._key("Austria")

    def test_similar_names_stay_different(self):
        assert self._key("Austria") != self._key("Australia")

    def test_former_states_not_merged(self):
        assert self._key("Yugoslavia") != self._key("Slovenia")

    def test_na_passes_through(self):
        assert self._key(None) is None
        assert self._key("99") == "99"


def test_birth_country_scored_by_exact_match():
    from corex_eval.evaluate import _country_code_names, _country_key
    from corex_eval.metrics.extraction import atomic_accuracy
    gold = ["17 = Belgium", "1 = Austria", "5 = Czech Republic"]
    pred = ["Belgium", "Australia", "Czechia"]
    names = _country_code_names(gold)
    r = atomic_accuracy([_country_key(p, names) for p in pred],
                        [_country_key(g, names) for g in gold])
    assert r["accuracy"] == pytest.approx(2 / 3)
