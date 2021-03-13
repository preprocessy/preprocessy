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
        ({"df": dataframe1}, ValueError),
        ({"df": array}, TypeError),
        ({"df": dataframe1, "type": [5]}, TypeError),
        (
            {"df": dataframe1, "type": "MinMaxScaler", "columns": "Distance"},
            TypeError,
        ),
        ({"df": dataframe1, "type": "nice"}, ArgumentsError),
        (
            {"df": dataframe1, "type": "MinMaxScaler", "columns": ["Area"]},
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
                "df": dataframe2,
                "type": "BinaryScaler",
                "columns": ["Negatives"],
                "is_combined": True,
                "critical_value": -10,
            }
        ),
        (
            {
                "df": dataframe2,
                "type": "BinaryScaler",
                "columns": ["Negatives"],
                "is_combined": True,
                "critical_value": -1,
            }
        ),
        {
            "df": dataframe2,
            "type": "BinaryScaler",
            "columns": ["Negatives"],
            "is_combined": True,
            "critical_value": -13000,
        },
    ],
)
def test_BinaryScaler_output(test_input):
    scaler = Scaler()
    scaler.execute(params=test_input)
    assert (
        test_input["df"]["Negatives"].values.any() == 1
        or test_input["df"]["Negatives"].values.any() == 0
    )
    assert not (
        test_input["df"]["Negatives"].between(0, 1, inclusive=False).any()
    )
    assert test_input["df"]["Negatives"][0] == 1


@pytest.mark.parametrize(
    "test_input",
    [
        (
            {
                "df": dataframe1,
                "type": "MinMaxScaler",
                "columns": ["Distance"],
                "is_combined": False,
            }
        ),
        (
            {
                "df": dataframe1,
                "type": "MinMaxScaler",
                "columns": ["Distance"],
                "is_combined": True,
            }
        ),
        (
            {
                "df": dataframe1,
                "type": "StandardScaler",
                "columns": ["Distance"],
                "is_combined": False,
            }
        ),
        (
            {
                "df": dataframe1,
                "type": "StandardScaler",
                "columns": ["Distance"],
                "is_combined": True,
            }
        ),
    ],
)
def test_MinMaxScaler_output(test_input):
    scaler = Scaler()
    scaler.execute(params=test_input)
    assert test_input["df"]["Distance"].values.all() >= 0
    assert test_input["df"]["Distance"].values.all() <= 1
