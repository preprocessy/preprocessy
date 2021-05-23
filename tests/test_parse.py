import pandas as pd
import pytest

from preprocessy.parse import Parser

ord_dict = {"Profession": {"Student": 1, "Teacher": 2, "HOD": 3}}
train_csv = pd.read_csv("datasets/encoding/testnew.csv")


def test_empty_df():
    params = {"target_label": "Price", "ord_dict": ord_dict}
    with pytest.raises(ValueError):
        parser = Parser()
        parser.parse_dataset(params=params)


def test_target_label_warning():
    train_csv = pd.read_csv("datasets/encoding/testnew.csv")
    params = {"train_df": train_csv, "ord_dict": ord_dict}
    with pytest.warns(UserWarning):
        parser = Parser()
        parser.parse_dataset(params=params)


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
        parser = Parser()
        parser.parse_dataset(params=params)


def test_parser():
    train_df = pd.DataFrame(
        {
            "A": [i for i in range(100)],
            "B": ["hello" if i % 2 == 0 else "bye" for i in range(100)],
        }
    )
    params = {"train_df": train_df, "target_label": "C"}
    parser = Parser()
    parser.parse_dataset(params=params)
    assert "B" in params["cat_cols"]
