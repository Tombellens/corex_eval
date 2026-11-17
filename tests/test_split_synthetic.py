"""
Split tests on synthetic data — run without COREX_DATA_DIR.
"""

import warnings

import pandas as pd
import pytest

from corex_eval.config import SPLIT_SEED, SPLIT_TEST_SIZE


@pytest.fixture(scope="session", autouse=True)
def require_data_dir():
    """Override the conftest skip: these tests need no data."""
    return


def _make_df(n_countries=30, per_country=60, seed=0):
    import random
    rng = random.Random(seed)
    rows = []
    for c in range(n_countries):
        for k in range(rng.randint(per_country // 2, per_country)):
            rows.append({"case_id": f"C{c:02d}_{k:03d}", "country_label": f"country_{c}"})
    rows.append({"case_id": "SINGLE_1", "country_label": "lonely"})
    rng.shuffle(rows)
    return pd.DataFrame(rows)


def _legacy_test_ids(df):
    """The split as implemented before the dev set existed (frozen copy)."""
    from collections import Counter
    from sklearn.model_selection import train_test_split
    ids = df["case_id"].astype(str).tolist()
    strata = df["country_label"].astype(str).tolist()
    counts = Counter(strata)
    keep = [counts[s] >= 2 for s in strata]
    ids = [i for i, k in zip(ids, keep) if k]
    strata = [s for s, k in zip(strata, keep) if k]
    _, test = train_test_split(
        ids, test_size=SPLIT_TEST_SIZE, random_state=SPLIT_SEED, stratify=strata
    )
    return set(test)


@pytest.fixture(scope="module")
def df():
    return _make_df()


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_test_ids_unchanged_by_dev_split(df):
    from corex_eval.split import get_test_ids
    assert get_test_ids(df) == _legacy_test_ids(df)


def test_three_splits_partition_the_data(df):
    from corex_eval.split import get_dev_ids, get_test_ids, get_train_ids
    train, dev, test = get_train_ids(df), get_dev_ids(df), get_test_ids(df)
    assert not (train & dev) and not (train & test) and not (dev & test)
    assert train | dev | test == set(df["case_id"])


def test_include_dev_returns_full_pool(df):
    from corex_eval.split import get_dev_ids, get_train_ids
    assert get_train_ids(df, include_dev=True) == get_train_ids(df) | get_dev_ids(df)


def test_dev_is_deterministic_and_order_independent(df):
    from corex_eval.split import get_dev_ids
    shuffled = df.sample(frac=1, random_state=7).reset_index(drop=True)
    assert get_dev_ids(df) == get_dev_ids(df)
    # Test ids depend on row order (legacy behaviour), so only compare dev
    # within the same train pool.
    from corex_eval.split import _carve_dev, _split
    train_ids, _ = _split(df)
    assert set(_carve_dev(df, train_ids)) == set(_carve_dev(shuffled, list(reversed(train_ids))))


def test_dev_is_stratified_by_country(df):
    from corex_eval.split import get_dev_ids
    dev = get_dev_ids(df)
    dev_countries = set(df[df["case_id"].isin(dev)]["country_label"])
    assert len(dev_countries) >= 29  # all multi-row countries represented


def test_singleton_country_stays_in_train(df):
    from corex_eval.split import get_dev_ids, get_test_ids, get_train_ids
    assert "SINGLE_1" in get_train_ids(df)
    assert "SINGLE_1" not in get_dev_ids(df) | get_test_ids(df)


def test_apply_eval_split_rejects_unknown(df):
    from corex_eval.split import apply_eval_split
    with pytest.raises(ValueError):
        apply_eval_split(df, "train")


def test_evaluate_refuses_dev_submission():
    from corex_eval import evaluate
    preds = pd.DataFrame({"case_id": ["x"], "birth_year": [1960]})
    with pytest.raises(ValueError, match="test-split"):
        evaluate(preds, task="extraction", variable="birth_year",
                 split="dev", submit=True, experiment_path="x.yaml")


def test_evaluate_rejects_unknown_split():
    from corex_eval import evaluate
    preds = pd.DataFrame({"case_id": ["x"], "birth_year": [1960]})
    with pytest.raises(ValueError, match="split must be"):
        evaluate(preds, task="extraction", variable="birth_year", split="train")
