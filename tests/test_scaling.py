import numpy as np
import pandas as pd
import pytest

from preprocessy.exceptions import ArgumentsError
from preprocessy.scaling import Scaler

dataframe1 = pd.read_csv("datasets/encoding/test2.csv")
dataframe2 = pd.read_csv("datasets/encoding/test.csv")
array = np.random.random((5, 5))


@pytest.mark.parametrize(
    "test_input, error",
    [
        ({}, ValueError),
        ({"train_df": dataframe1}, TypeError),
        ({"train_df": array}, TypeError),
        ({"train_df": dataframe1, "type": [5]}, TypeError),
        (
            {"train_df": dataframe1, "type": "MinMaxScaler", "columns": "Distance"},
            TypeError,
        ),
        ({"train_df": dataframe1, "type": "nice"}, ArgumentsError),
        (
            {"train_df": dataframe1, "type": "MinMaxScaler", "columns": ["Area"]},
            ArgumentsError,
        ),
    ],
)
def test_incorrect_input_type(test_input, error):
    with pytest.raises(error):
        scaler = Scaler()
        scaler.execute(params=test_input)


@pytest.mark.parametrize(
    "test_input",
    [
        (
            {
                "train_df": dataframe2,
                "type": "BinaryScaler",
                "columns": ["Negatives"],
                "is_combined": True,
                "threshold": {"Negatives": -10},
                "target_col": "Test",
            }
        ),
        (
            {
                "train_df": dataframe2,
                "type": "BinaryScaler",
                "columns": ["Negatives"],
                "is_combined": True,
                "threshold": {"Negatives": -1},
                "target_col": "Test",
            }
        ),
        {
            "train_df": dataframe2,
            "type": "BinaryScaler",
            "columns": ["Negatives"],
            "is_combined": True,
            "threshold": {"Negatives": -1300},
            "target_col": "Test",
        },
    ],
)
def test_BinaryScaler_output(test_input):
    scaler = Scaler()
    scaler.execute(params=test_input)
    assert (
        test_input["train_df"]["Negatives"].values.any() == 1
        or test_input["train_df"]["Negatives"].values.any() == 0
    )
    assert not (
        test_input["train_df"]["Negatives"].between(0, 1, inclusive=False).any()
    )
    if test_input["threshold"]["Negatives"] != -1:
        assert test_input["train_df"]["Negatives"][0] == 1
    else:
        assert test_input["train_df"]["Negatives"][0] == 0


@pytest.mark.parametrize(
    "test_input",
    [
        (
            {
                "train_df": dataframe1,
                "type": "MinMaxScaler",
                "columns": ["Distance"],
                "is_combined": False,
                "target_col": "Capitals",
                "categorical_columns": ["Capitals", "Other Capitals"],
            }
        ),
        (
            {
                "train_df": dataframe1,
                "type": "MinMaxScaler",
                "columns": ["Distance"],
                "is_combined": True,
                "target_col": "Capitals",
                "categorical_columns": ["Capitals", "Other Capitals"],
            }
        ),
        (
            {
                "train_df": dataframe2,
                "type": "MinMaxScaler",
                "columns": ["Negatives"],
                "is_combined": False,
                "target_col": "Test",
                "categorical_columns": ["Price", "Profession", "Date"],
            }
        ),
        (
            {
                "train_df": dataframe2,
                "type": "MinMaxScaler",
                "columns": ["Negatives"],
                "is_combined": True,
                "target_col": "Test",
                "categorical_columns": ["Price", "Profession", "Date"],
            }
        ),
    ],
)
def test_MinMaxScaler_output(test_input):
    scaler = Scaler()
    scaler.execute(params=test_input)
    if "Distance" in test_input["train_df"].keys():
        assert test_input["train_df"]["Distance"].values.all() >= 0
        assert test_input["train_df"]["Distance"].values.all() <= 1
    else:
        assert test_input["train_df"]["Negatives"].values.all() >= 0
        assert test_input["train_df"]["Negatives"].values.all() <= 1


@pytest.mark.parametrize(
    "test_input",
    [
        (
            {
                "train_df": dataframe1,
                "type": "StandardScaler",
                "columns": ["Distance"],
                "is_combined": False,
                "target_col": "Other Capitals",
                "categorical_columns": ["Capitals", "Other Capitals"],
            }
        ),
        (
            {
                "train_df": dataframe1,
                "type": "StandardScaler",
                "columns": ["Distance"],
                "is_combined": True,
                "target_col": "Capitals",
                "categorical_columns": ["Capitals", "Other Capitals"],
            }
        ),
        (
            {
                "train_df": dataframe2,
                "type": "StandardScaler",
                "columns": ["Negatives"],
                "is_combined": False,
                "target_col": "Test",
                "categorical_columns": ["Price", "Profession", "Date"],
            }
        ),
        (
            {
                "train_df": dataframe2,
                "type": "StandardScaler",
                "columns": ["Negatives"],
                "is_combined": True,
                "target_col": "Test",
                "categorical_columns": ["Price", "Profession", "Date"],
            }
        ),
    ],
)
def test_StandardScaler_output(test_input):
    scaler = Scaler()
    scaler.execute(params=test_input)
    if "Distance" in test_input["train_df"].keys():
        assert round(test_input["train_df"]["Distance"][0], 5) == 1.08006
    else:
        assert round(test_input["train_df"]["Negatives"][0], 5) == 0.57771
