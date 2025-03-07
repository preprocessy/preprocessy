import numpy as np
import pandas as pd
import pytest
from preprocessy.exceptions import ArgumentsError
from preprocessy.missing_data import NullValuesHandler

dataframe1 = pd.read_csv("datasets/encoding/test2.csv")
dataframe2 = pd.read_csv("datasets/encoding/testnew.csv")
titanic = pd.read_csv("datasets/titanic.csv")
array = np.random.random((5, 5))


def test_null_dataframe():
    handler = NullValuesHandler()
    with pytest.raises(ValueError):
        handler.execute({})


@pytest.mark.parametrize(
    "error, test_input",
    [
        (TypeError, {"train_df": array}),
        (TypeError, {"train_df": dataframe2, "drop_cols": "nice"}),
        (TypeError, {"train_df": dataframe1, "fill_missing": 3}),
        (TypeError, {"train_df": dataframe1, "fill_values": [5]}),
        (TypeError, {"train_df": dataframe1, "drop_cols": True, "cat_cols": 4}),
        (
            TypeError,
            {
                "train_df": titanic,
                "cat_cols": ["Pclass", "Sex", "Parch", "Embarked"],
                "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
                "fill_missing": ["mean", "Age"],
            },
        ),
        (
            TypeError,
            {
                "train_df": titanic,
                "cat_cols": ["Pclass", "Sex", "Parch", "Embarked"],
                "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
                "fill_missing": {"mean": "Age"},
            },
        ),
        (
            TypeError,
            {
                "train_df": titanic,
                "cat_cols": ["Pclass", "Sex", "Parch", "Embarked"],
                "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
                "fill_missing": {"mean": ["Name"]},
            },
        ),
        (ArgumentsError, {"train_df": dataframe1, "fill_missing": {"sum": []}}),
        (
            ArgumentsError,
            {
                "train_df": dataframe1,
                "drop_cols": ["Yolo"],
                "cat_cols": ["xyz"],
            },
        ),
        (
            ArgumentsError,
            {"train_df": dataframe1, "fill_values": {"Label": 10}},
        ),
        (ArgumentsError, {"train_df": dataframe1, "test_df": dataframe2}),
        (
            ArgumentsError,
            {
                "train_df": titanic,
                "cat_cols": ["Pclass", "Sex", "Parch", "Embarked"],
                "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
                "fill_missing": {"mean": ["Age"], "median": []},
            },
        ),
        (
            ArgumentsError,
            {
                "train_df": titanic,
                "cat_cols": ["Pclass", "Sex", "Parch", "Embarked"],
                "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
                "fill_missing": {"mean": ["Car"]},
            },
        ),
    ],
)
def test_incorrect_input_type(error, test_input):
    handler = NullValuesHandler()
    with pytest.raises(error):
        handler.execute(params=test_input)


# to test output
@pytest.mark.parametrize(
    "test_input1",
    [
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_values": {"Other Capitals": "0"},
            "cat_cols": ["Other Capitals"],
        },
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_missing": {"mean": ["Distance"]},
        },
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_values": {"Distance": 0},
        },
    ],
)
def test_categorical_drops(test_input1):
    train_row_count = (test_input1["train_df"].shape)[0]
    test_row_count = (test_input1["test_df"].shape)[0]
    handler = NullValuesHandler()
    handler.execute(params=test_input1)
    assert "0" not in test_input1["train_df"]["Other Capitals"]
    assert (test_input1["train_df"].shape)[0] <= train_row_count
    assert "0" not in test_input1["test_df"]["Other Capitals"]
    assert (test_input1["test_df"].shape)[0] <= test_row_count


@pytest.mark.parametrize(
    "test_input2",
    [
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_values": {"Other Capitals": "0"},
            "cat_cols": ["Capitals"],
            "replace_cat_nulls": "xyz",
        },
    ],
)
def test_categorical_replace(test_input2):
    test_input2["train_df"] = pd.read_csv("datasets/encoding/test2.csv")
    test_input2["test_df"] = pd.read_csv("datasets/encoding/test2.csv")
    handler = NullValuesHandler()
    handler.execute(params=test_input2)
    assert (
        len(
            test_input2["train_df"][test_input2["train_df"].Capitals == "xyz"][
                "Capitals"
            ]
        )
        == 8
    )
    assert (
        len(
            test_input2["test_df"][test_input2["test_df"].Capitals == "xyz"][
                "Capitals"
            ]
        )
        == 8
    )


@pytest.mark.parametrize(
    "test_input3",
    [
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_values": {"Other Capitals": "0"},
            "cat_cols": ["Other Capitals"],
        },
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_missing": {"mean": ["Distance"]},
        },
        {
            "train_df": dataframe1,
            "test_df": dataframe1,
            "fill_values": {"Distance": 0},
        },
    ],
)
def test_output(test_input3):
    handler = NullValuesHandler()
    handler.execute(params=test_input3)
    assert not test_input3["train_df"].isnull().any()["Distance"]
    assert not test_input3["test_df"].isnull().any()["Distance"]


# for dropping list of columns passed by user
def test_drop_col():
    params = {
        "train_df": dataframe1,
        "test_df": dataframe1,
        "drop_cols": ["Distance"],
    }
    handler = NullValuesHandler()
    handler.execute(params=params)
    assert "Distance" not in params["train_df"].columns
    assert "Distance" not in params["test_df"].columns


def test_auto():
    dataframe1 = pd.read_csv("datasets/encoding/test2.csv")
    params = {"train_df": dataframe1, "test_df": dataframe1}
    r = (params["train_df"].shape)[0]
    rt = (params["test_df"].shape)[0]
    handler = NullValuesHandler()
    handler.execute(params=params)
    r_new = (params["train_df"].shape)[0]
    rt_new = (params["test_df"].shape)[0]
    assert r_new != r
    assert rt_new != rt


def test_mulitple():
    params = {
        "train_df": titanic,
        "cat_cols": ["Pclass", "Sex", "Parch", "Embarked"],
        "drop_cols": ["PassengerId", "Name", "Ticket", "Cabin"],
        "fill_missing": {"mean": ["Age"], "median": ["Fare"]},
    }
    handler = NullValuesHandler()
    handler.execute(params=params)
    assert params["train_df"].shape == (889, 8)
