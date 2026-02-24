"""
test_gold.py
============
Tests for corex_eval.gold — loading, cleaning, and reshaping.

Location in project:
    corex_eval/
    └── tests/
        └── test_gold.py
"""

import pytest
import pandas as pd


class TestLoadGold:

    def test_returns_dataframe(self, gold_df):
        assert isinstance(gold_df, pd.DataFrame)

    def test_has_case_id_column(self, gold_df):
        assert "case_id" in gold_df.columns

    def test_no_empty_case_ids(self, gold_df):
        empty = gold_df["case_id"].astype(str).str.strip().eq("")
        assert empty.sum() == 0, f"{empty.sum()} rows with empty case_id"

    def test_no_missing_type_rows(self, gold_df):
        """Incomplete records must be dropped during loading."""
        if "missing_type" in gold_df.columns:
            flagged = gold_df["missing_type"].astype(str).str.strip().ne("")
            assert flagged.sum() == 0, f"{flagged.sum()} incomplete records remain"

    def test_birth_year_is_nullable_int(self, gold_df):
        assert str(gold_df["birth_year"].dtype) == "Int64", (
            f"Expected Int64, got {gold_df['birth_year'].dtype}"
        )

    def test_all_year_columns_are_numeric(self, gold_df):
        year_cols = [c for c in gold_df.columns if "year" in c.lower()]
        for col in year_cols:
            assert str(gold_df[col].dtype) == "Int64", (
                f"'{col}' has dtype {gold_df[col].dtype}, expected Int64"
            )

    def test_no_duplicate_case_ids(self, gold_df):
        dupes = gold_df["case_id"].duplicated()
        assert dupes.sum() == 0, (
            f"{dupes.sum()} duplicate case_ids: "
            f"{gold_df[dupes]['case_id'].tolist()[:5]}"
        )

    def test_row_count_in_expected_range(self, gold_df):
        """Belgium has 201 raw rows; after cleaning expect ~180."""
        assert 100 < len(gold_df) <= 201, f"Unexpected row count: {len(gold_df)}"

    def test_loading_twice_is_identical(self, gold_df):
        """load_gold() must be deterministic."""
        import warnings
        from corex_eval.gold import load_gold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df2 = load_gold()
        assert list(df2["case_id"]) == list(gold_df["case_id"])

    def test_missing_file_raises_clear_error(self, tmp_path):
        from corex_eval.gold import load_gold
        with pytest.raises(FileNotFoundError, match="corex_gold.csv"):
            load_gold(path=tmp_path / "nonexistent.csv")


class TestGetCareerSpells:

    def test_returns_dataframe(self, gold_df):
        from corex_eval.gold import get_career_spells
        spells = get_career_spells(gold_df)
        assert isinstance(spells, pd.DataFrame)

    def test_has_required_columns(self, gold_df):
        from corex_eval.gold import get_career_spells
        spells = get_career_spells(gold_df)
        for col in ["case_id", "spell_index", "career_start_year",
                    "career_end_year", "career_position"]:
            assert col in spells.columns, f"Missing column: {col}"

    def test_no_empty_position_rows(self, gold_df):
        from corex_eval.gold import get_career_spells
        spells = get_career_spells(gold_df)
        empty = spells["career_position"].astype(str).str.strip().eq("")
        assert empty.sum() == 0, f"{empty.sum()} rows with empty career_position"

    def test_spell_count_is_reasonable(self, gold_df):
        from corex_eval.gold import get_career_spells
        spells = get_career_spells(gold_df)
        assert len(spells) > 100, f"Too few spells: {len(spells)}"

    def test_spell_indices_start_at_one(self, gold_df):
        from corex_eval.gold import get_career_spells
        spells = get_career_spells(gold_df)
        assert spells["spell_index"].min() == 1

    def test_no_duplicate_spell_indices_per_case(self, gold_df):
        from corex_eval.gold import get_career_spells
        spells = get_career_spells(gold_df)
        dupes = spells.duplicated(subset=["case_id", "spell_index"])
        assert dupes.sum() == 0, f"{dupes.sum()} duplicate (case_id, spell_index) pairs"

    def test_all_case_ids_in_gold(self, gold_df):
        """Every spell's case_id must exist in the gold dataframe."""
        from corex_eval.gold import get_career_spells
        spells   = get_career_spells(gold_df)
        gold_ids = set(gold_df["case_id"].astype(str))
        spell_ids = set(spells["case_id"].astype(str))
        orphans = spell_ids - gold_ids
        assert len(orphans) == 0, f"Spells with unknown case_ids: {orphans}"
