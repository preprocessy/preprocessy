import pandas as pd
import pytest

from preprocessy.exceptions import ArgumentsError
from preprocessy.outliers import HandleOutlier

train_df = pd.read_csv("datasets/encoding/test2.csv")


@pytest.mark.parametrize(
    "test_input",
    [
        {},
        {
            "train_df": train_df,
            "cat_cols": ["Capitals"],
            "target_label": "Other Capitals",
            "first_quartile": 1.3,
        },
        {
            "train_df": train_df,
            "cat_cols": ["Capitals"],
            "target_label": "Other Capitals",
            "first_quartile": 0.3,
            "third_quartile": 1.54,
        },
        {
            "train_df": train_df,
            "cat_cols": ["Capitals"],
            "target_label": "Other Capitals",
            "first_quartile": 0.7,
            "third_quartile": 0.3,
        },
    ],
)
def test_incorrect_input_value(test_input):
    with pytest.raises(ValueError):
        outlier = HandleOutlier()
        outlier.handle_outliers(params=test_input)


# test for none operation
def test_false_arguments():
    with pytest.warns(UserWarning):
        outlier = HandleOutlier()
        outlier.handle_outliers(
            params={
                "train_df": train_df,
                "cat_cols": ["Capitals", "Other Capitals"],
                "remove_outliers": False,
                "replace": False,
            }
        )


# test for multiple operation
def test_true_arguments():
    with pytest.raises(ArgumentsError):
        outlier = HandleOutlier()
        outlier.handle_outliers(
            params={
                "train_df": train_df,
                "cat_cols": ["Capitals", "Other Capitals"],
                "replace": True,
            }
        )


@pytest.mark.parametrize(
    "test_input",
    [
        {"train_df": 5},
        {"train_df": train_df, "cat_cols": "Capitals"},
        {"train_df": train_df, "cat_cols": [5]},
        {
            "train_df": train_df,
            "cat_cols": ["Capitals", "Other Capitals"],
            "remove_outliers": "True",
        },
        {
            "train_df": train_df,
            "cat_cols": ["Capitals", "Other Capitals"],
            "remove_outliers": False,
            "replace": "True",
        },
        {
            "train_df": train_df,
            "cat_cols": ["Capitals", "Other Capitals"],
            "first_quartile": "0.24",
        },
        {
            "train_df": train_df,
            "cat_cols": ["Capitals", "Other Capitals"],
            "first_quartile": 0.24,
            "third_quartile": "0.76",
        },
    ],
)
def test_incorrect_input_type(test_input):
    with pytest.raises(TypeError):
        outlier = HandleOutlier()
        outlier.handle_outliers(params=test_input)


def test_incorrect_input_key():
    with pytest.raises(KeyError):
        outlier = HandleOutlier()
        outlier.handle_outliers(
            params={"train_df": train_df, "cols": ["Place"]}
        )


@pytest.mark.parametrize(
    "test_input",
    [{"train_df": train_df, "cat_cols": ["Capitals", "Other Capitals"]}],
)
def test_removeoutliers_output(test_input):
    outlier = HandleOutlier()
    outlier.handle_outliers(params=test_input)
    assert test_input["train_df"].shape != train_df.shape


@pytest.mark.parametrize(
    "test_input",
    [
        {
            "train_df": train_df,
            "cat_cols": ["Capitals"],
            "target_label": "Other Capitals",
            "remove_outliers": False,
            "replace": True,
        }
    ],
)
def test_replace_output(test_input):
    outlier = HandleOutlier()
    outlier.handle_outliers(params=test_input)
    assert test_input["train_df"].shape == train_df.shape
    assert -999 in test_input["train_df"]["Distance"].values
    assert "-999" not in test_input["train_df"]["Capitals"].values
    assert "-999" not in test_input["train_df"]["Other Capitals"].values


@pytest.mark.parametrize(
    "test_input",
    [
        {
            "train_df": train_df,
            "test_df": train_df,
            "cat_cols": ["Capitals", "Other Capitals"],
            "remove_outliers": False,
            "replace": True,
        }
    ],
)
def test_all(test_input):
    outlier = HandleOutlier()
    outlier.handle_outliers(params=test_input)
    assert -999 in test_input["train_df"]["Distance"].values
    assert -999 in test_input["test_df"]["Distance"].values
    assert test_input["train_df"].equals(test_input["test_df"])
