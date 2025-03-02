import numpy as np
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
    outlier = HandleOutlier()
    with pytest.raises(ValueError):
        outlier.handle_outliers(params=test_input)


# test for none operation
def test_false_arguments():
    outlier = HandleOutlier()
    with pytest.warns(UserWarning):
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
    outlier = HandleOutlier()
    with pytest.raises(ArgumentsError):
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
    outlier = HandleOutlier()
    with pytest.raises(TypeError):
        outlier.handle_outliers(params=test_input)


def test_incorrect_input_key():
    outlier = HandleOutlier()
    with pytest.raises(KeyError):
        outlier.handle_outliers(
            params={"train_df": train_df, "out_cols": ["Place"]}
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


def test_outlier_replace_only_in_numeric_columns():
    a = np.random.rand(
        100,
    )
    a[95:] = 1504360
    b = [0 if i % 2 == 0 else 1 for i in range(100)]
    b = np.asarray(b)
    b[70:] = 42
    data = {"A": a, "B": b}
    sample_df = pd.DataFrame(data)
    params = {
        "train_df": sample_df,
        "cat_cols": ["B"],
        "remove_outliers": False,
        "replace": True,
    }
    outlier = HandleOutlier()
    outlier.handle_outliers(params=params)
    assert -999 in params["train_df"]["A"].values[95:]
    assert (params["train_df"]["B"].compare(sample_df["B"])).empty
