"""
test_inputs.py
==============
Tests for corex_eval.inputs — load_inputs() and load_training_data().

Location in project:
    corex_eval/
    └── tests/
        └── test_inputs.py
"""

import warnings
import pytest
import pandas as pd


class TestLoadInputsCollection:

    def test_returns_dataframe(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="collection")
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="collection")
        for col in ["case_id", "name_first", "name_last", "job_title", "country_label"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_gold_url_column_not_exposed(self):
        """cv_links is the gold target — must not appear in inputs."""
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="collection")
        assert "cv_links" not in df.columns

    def test_contains_only_test_rows(self, test_df):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="collection")
        input_ids = set(df["case_id"].astype(str))
        test_ids  = set(test_df["case_id"].astype(str))
        assert input_ids.issubset(test_ids), (
            f"Input contains non-test IDs: {input_ids - test_ids}"
        )

    def test_sorted_by_case_id(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="collection")
        ids = df["case_id"].astype(str).tolist()
        assert ids == sorted(ids)

    def test_no_empty_case_ids(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="collection")
        empty = df["case_id"].astype(str).str.strip().eq("")
        assert empty.sum() == 0


class TestLoadInputsExtraction:

    def test_returns_dataframe(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="extraction")
        assert isinstance(df, pd.DataFrame)

    def test_has_case_id_and_cv_local(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="extraction")
        assert "case_id"  in df.columns
        assert "cv_local" in df.columns

    def test_no_empty_cv_local_rows(self):
        """Rows where cv_local is empty must be dropped at load time."""
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="extraction")
        empty = df["cv_local"].astype(str).str.strip().isin(["", "nan", "<NA>"])
        assert empty.sum() == 0, f"{empty.sum()} rows have empty cv_local"

    def test_gold_labels_not_exposed(self):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="extraction")
        for col in ["birth_year", "birth_place", "sex", "edu_degree"]:
            assert col not in df.columns, f"Gold column '{col}' leaked into inputs"

    def test_contains_only_test_rows(self, test_df):
        from corex_eval import load_inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_inputs(task="extraction")
        input_ids = set(df["case_id"].astype(str))
        test_ids  = set(test_df["case_id"].astype(str))
        assert input_ids.issubset(test_ids)


class TestLoadInputsAnnotation:

    def test_raises_when_variable_missing(self):
        from corex_eval import load_inputs
        with pytest.raises(ValueError, match="variable"):
            load_inputs(task="annotation")

    def test_raises_on_unknown_variable(self):
        from corex_eval import load_inputs
        with pytest.raises(ValueError):
            load_inputs(task="annotation", variable="not_a_real_variable")

    def test_raises_when_silver_file_missing(self):
        """Clear FileNotFoundError until silver standard is built."""
        from corex_eval import load_inputs
        with pytest.raises(FileNotFoundError, match="corex_silver.csv"):
            load_inputs(task="annotation", variable="career_position")

    def test_edu_degree_is_valid_annotation_variable(self):
        """edu_degree must not be rejected as an unknown annotation variable."""
        from corex_eval import load_inputs
        try:
            load_inputs(task="annotation", variable="edu_degree")
        except (FileNotFoundError, ValueError) as exc:
            # FileNotFoundError  → silver not yet built (ok)
            # ValueError         → silver exists but edu columns not yet built (ok)
            # Either way, the error must NOT say "Unknown annotation variable"
            assert "Unknown annotation variable" not in str(exc), (
                f"edu_degree was not recognised as a valid annotation variable: {exc}"
            )

    def test_uni_subject_is_valid_annotation_variable(self):
        """uni_subject must not be rejected as an unknown annotation variable."""
        from corex_eval import load_inputs
        try:
            load_inputs(task="annotation", variable="uni_subject")
        except (FileNotFoundError, ValueError) as exc:
            assert "Unknown annotation variable" not in str(exc), (
                f"uni_subject was not recognised as a valid annotation variable: {exc}"
            )


class TestLoadInputsValidation:

    def test_rejects_unknown_task(self):
        from corex_eval import load_inputs
        with pytest.raises(ValueError, match="unknown_task"):
            load_inputs(task="unknown_task")


class TestLoadTrainingData:

    def test_returns_dataframe(self):
        from corex_eval import load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_training_data(task="extraction", features=["birth_year"])
        assert isinstance(df, pd.DataFrame)

    def test_always_includes_case_id(self):
        from corex_eval import load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_training_data(task="extraction", features=["birth_year"])
        assert "case_id" in df.columns

    def test_returns_requested_features(self):
        from corex_eval import load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_training_data(
                task="extraction",
                features=["birth_year", "sex"]
            )
        assert "birth_year" in df.columns
        assert "sex"        in df.columns

    def test_contains_only_train_rows(self, train_df, test_df):
        from corex_eval import load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = load_training_data(task="extraction", features=["birth_year"])
        result_ids = set(df["case_id"].astype(str))
        test_ids   = set(test_df["case_id"].astype(str))
        assert len(result_ids & test_ids) == 0, (
            f"Training data contains test IDs: {result_ids & test_ids}"
        )

    def test_train_and_test_inputs_are_disjoint(self, test_df):
        from corex_eval import load_inputs, load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_inp  = load_inputs(task="collection")
            train_inp = load_training_data(task="collection", features=[])
        test_ids  = set(test_inp["case_id"].astype(str))
        train_ids = set(train_inp["case_id"].astype(str))
        assert len(test_ids & train_ids) == 0

    def test_warns_on_unknown_feature(self):
        from corex_eval import load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            with pytest.warns(UserWarning, match="not found"):
                load_training_data(
                    task="extraction",
                    features=["birth_year", "this_column_does_not_exist_xyz"]
                )

    def test_train_is_larger_than_test(self, test_df):
        from corex_eval import load_training_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train = load_training_data(task="extraction", features=[])
        assert len(train) > len(test_df)
