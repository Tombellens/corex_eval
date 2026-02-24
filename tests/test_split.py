"""
test_split.py
=============
Tests for corex_eval.split — deterministic, stratified 80/20 split.

Location in project:
    corex_eval/
    └── tests/
        └── test_split.py
"""

import pytest


class TestSplitProportions:

    def test_test_is_roughly_20_percent(self, gold_df, test_df):
        ratio = len(test_df) / len(gold_df)
        assert abs(ratio - 0.20) < 0.02, f"Test ratio was {ratio:.3f}"

    def test_train_is_roughly_80_percent(self, gold_df, train_df):
        ratio = len(train_df) / len(gold_df)
        assert abs(ratio - 0.80) < 0.02, f"Train ratio was {ratio:.3f}"

    def test_sizes_sum_to_total(self, gold_df, test_df, train_df):
        assert len(test_df) + len(train_df) == len(gold_df)


class TestSplitDiSjointness:

    def test_no_overlap_between_splits(self, test_df, train_df):
        test_ids  = set(test_df["case_id"].astype(str))
        train_ids = set(train_df["case_id"].astype(str))
        overlap   = test_ids & train_ids
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping IDs: {overlap}"

    def test_union_equals_full_dataset(self, gold_df, test_df, train_df):
        all_ids   = set(gold_df["case_id"].astype(str))
        test_ids  = set(test_df["case_id"].astype(str))
        train_ids = set(train_df["case_id"].astype(str))
        assert test_ids | train_ids == all_ids


class TestSplitReproducibility:

    def test_split_is_deterministic(self, gold_df, test_df):
        """Running the split twice must produce the same test set."""
        import warnings
        from corex_eval.split import apply_test_split
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test2 = apply_test_split(gold_df)
        assert set(test2["case_id"]) == set(test_df["case_id"])

    def test_train_split_is_deterministic(self, gold_df, train_df):
        import warnings
        from corex_eval.split import apply_train_split
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train2 = apply_train_split(gold_df)
        assert set(train2["case_id"]) == set(train_df["case_id"])

    def test_does_not_mutate_input(self, gold_df):
        import warnings
        from corex_eval.split import apply_test_split
        original_len = len(gold_df)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = apply_test_split(gold_df)
        assert len(gold_df) == original_len


class TestGetIds:

    def test_get_test_ids_returns_set(self, gold_df):
        from corex_eval.split import get_test_ids
        ids = get_test_ids(gold_df)
        assert isinstance(ids, set)
        assert len(ids) > 0

    def test_get_train_ids_returns_set(self, gold_df):
        from corex_eval.split import get_train_ids
        ids = get_train_ids(gold_df)
        assert isinstance(ids, set)
        assert len(ids) > 0

    def test_get_test_ids_matches_apply_test_split(self, gold_df, test_df):
        from corex_eval.split import get_test_ids
        ids = get_test_ids(gold_df)
        assert ids == set(test_df["case_id"].astype(str))

    def test_test_and_train_ids_are_disjoint(self, gold_df):
        from corex_eval.split import get_test_ids, get_train_ids
        assert len(get_test_ids(gold_df) & get_train_ids(gold_df)) == 0
