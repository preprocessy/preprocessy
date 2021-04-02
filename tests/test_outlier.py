import pandas as pd
import pytest

from preprocessy.exceptions import ArgumentsError
from preprocessy.outliers import HandleOutlier

train_df = pd.read_csv("datasets/encoding/test2.csv")


class TestOutlier:
    # test for empty input
    def test_empty_df(self):
        with pytest.raises(ValueError):
            outlier = HandleOutlier()
            outlier.handle_outliers(params={})

    # test for none operation
    def test_false_arguments(self):
        with pytest.warns(UserWarning):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={
                    "train_df": train_df,
                    "cols": ["Distance"],
                    "remove_outliers": False,
                    "replace": False,
                }
            )

    # test for multiple operation
    def test_true_arguments(self):
        with pytest.raises(ArgumentsError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={
                    "train_df": train_df,
                    "cols": ["Distance"],
                    "replace": True,
                }
            )

    def test_incorrect_input_type(self):
        # for dataframe
        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(params={"train_df": 5})

        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={"train_df": train_df, "cols": "Distance"}
            )

        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(params={"train_df": train_df, "cols": [5]})

        with pytest.raises(KeyError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={"train_df": train_df, "cols": ["Place"]}
            )

        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={
                    "train_df": train_df,
                    "cols": ["Distance"],
                    "remove_outliers": "True",
                }
            )

        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={
                    "train_df": train_df,
                    "cols": ["Distance"],
                    "replace": "True",
                }
            )

        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={
                    "train_df": train_df,
                    "cols": ["Distance"],
                    "first_quartile": "0.24",
                }
            )

        with pytest.raises(TypeError):
            outlier = HandleOutlier()
            outlier.handle_outliers(
                params={
                    "train_df": train_df,
                    "cols": ["Distance"],
                    "first_quartile": 0.24,
                    "third_quartile": "0.76",
                }
            )

    def test_output(self):
        params = {"train_df": train_df, "cols": ["Distance"]}
        outlier = HandleOutlier()
        outlier.handle_outliers(params=params)
        assert params["train_df"].shape != train_df.shape

        params = {
            "train_df": train_df,
            "cols": ["Distance"],
            "remove_outliers": False,
            "replace": True,
        }
        outlier = HandleOutlier()
        outlier.handle_outliers(params=params)
        assert params["train_df"].shape == train_df.shape
        assert -999 in params["train_df"]["Distance"].values
