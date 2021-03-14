import numpy as np
import pandas as pd
import pytest

from preprocessy.exceptions import ArgumentsError
from preprocessy.handlenullvalues import NullValuesHandler

dataframe1 = pd.read_csv("datasets/encoding/test2.csv")
dataframe2 = pd.read_csv("datasets/encoding/testnew.csv")
array = np.random.random((5, 5))


def test_null_dataframe():
    with pytest.raises(ValueError):
        handler = NullValuesHandler()
        handler.execute({})


def test_none_args():
    with pytest.raises(ArgumentsError):
        handler = NullValuesHandler()
        handler.execute({"df": dataframe1})


@pytest.mark.parametrize(
    "test_input",
    [
        {"df": dataframe1, "drop": True, "fill_missing": "mean"},
        {
            "df": dataframe1,
            "drop": True,
            "fill_values": {"Test": "Tata"},
        },
        {
            "df": dataframe1,
            "drop": True,
            "fill_values": {"Test": "Tata"},
        },
        {
            "df": dataframe1,
            "fill_missing": "mean",
            "fill_values": {"Test": "Tata"},
        },
    ],
)
def test_multiple_args(test_input):
    with pytest.raises(ArgumentsError):
        handler = NullValuesHandler()
        handler.execute(params=test_input)


@pytest.mark.parametrize(
    "error, test_input",
    [
        (TypeError, {"df": array}),
        (TypeError, {"df": dataframe2, "drop": "nice"}),
        (TypeError, {"df": dataframe1, "fill_missing": 3}),
        (TypeError, {"df": dataframe1, "fill_values": [5]}),
        (ArgumentsError, {"df": dataframe1, "fill_missing": "sum"}),
        (ArgumentsError, {"df": dataframe1, "fill_values": {"Label": 10}}),
    ],
)
def test_incorrect_input_type(error, test_input):
    with pytest.raises(error):
        handler = NullValuesHandler()
        handler.execute(params=test_input)


# to test output
@pytest.mark.parametrize(
    "test_input1",
    [
        {"df": dataframe1, "drop": True},
        {"df": dataframe1, "fill_missing": "mean"},
        {"df": dataframe1, "fill_values": {"Distance": 0}},
    ],
)
def test_output(test_input1):
    handler = NullValuesHandler()
    handler.execute(params=test_input1)
    assert not test_input1["df"].isnull().any()["Distance"]


# for dropping list of columns passed by user
def test_drop_col():
    params = {"df": dataframe1, "drop": True, "column_list": ["Distance"]}
    handler = NullValuesHandler()
    handler.execute(params=params)
    assert "Distance" not in params["df"].columns
