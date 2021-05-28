from collections import Counter

import pandas as pd
import pytest

from preprocessy.encoding import Encoder

ord_dict = {"Profession": {"Student": 1, "Teacher": 2, "HOD": 3}}


# test for empty input
def test_empty_df():
    params = {"target_label": "Price", "ord_dict": ord_dict}
    with pytest.raises(ValueError):
        encoder = Encoder()
        encoder.encode(params=params)


# test for warning
def test_target_label_warning():
    train_csv = pd.read_csv("datasets/encoding/testnew.csv")
    params = {"train_df": train_csv, "ord_dict": ord_dict}
    with pytest.warns(UserWarning):
        encoder = Encoder()
        encoder.encode(params=params)


# test ordinal encoding
def test_mapping():
    train_csv = pd.read_csv("datasets/encoding/testnew.csv")
    train_csv.drop(["Price"], axis=1, inplace=True)
    params = {
        "train_df": train_csv,
        "target_label": "Price",
        "ord_dict": ord_dict,
    }
    encoder = Encoder()
    encoder.encode(params=params)
    assert params["train_df"]["ProfessionEncoded"].nunique() == 3
    assert params["train_df"]["ProfessionEncoded"][2] == 3
    assert Counter(params["ord_dict"]["Profession"].values()) == Counter(
        params["train_df"]["ProfessionEncoded"].unique()
    )


# test for empty mapping
def test_empty_weight_mapping():
    train_csv = pd.read_csv("datasets/encoding/testnew.csv")
    train_csv.drop(["Price"], axis=1, inplace=True)
    ord_dict1 = ord_dict.copy()
    ord_dict1["Size"] = None
    params = {
        "train_df": train_csv,
        "target_label": "Price",
        "ord_dict": ord_dict1,
    }
    with pytest.raises(ValueError):
        encoder = Encoder()
        encoder.encode(params=params)


# test for one-hot encoding
def test_one_hot_encoding():
    train_csv = pd.read_csv("datasets/encoding/testnew.csv")
    params = {
        "train_df": train_csv,
        "target_label": "Price",
        "cat_cols": ["Test", "Labels"],
        "ord_dict": ord_dict,
        "one_hot": True,
    }
    encoder = Encoder()
    encoder.encode(params=params)
    assert "Test_Tata" in params["train_df"].columns
    assert params["train_df"]["Test_Tata"][1] == 1


def test_ignore_cat_col():
    train_csv = pd.read_csv("datasets/encoding/testnew.csv")
    params = {
        "train_df": train_csv,
        "target_label": "Price",
        "cat_cols": ["Profession"],
        "ord_dict": ord_dict,
        "one_hot": True,
    }
    encoder = Encoder()
    encoder.encode(params=params)
    assert "Profession_HOD" not in params["train_df"].columns


def test_parser():
    train_df = pd.DataFrame(
        {
            "A": [i for i in range(100)],
            "B": ["hello" if i % 2 == 0 else "bye" for i in range(100)],
        }
    )
    params = {"train_df": train_df, "target_label": "C"}
    encoder = Encoder()
    encoder.encode(params=params)
    assert "B" in params["cat_cols"]
