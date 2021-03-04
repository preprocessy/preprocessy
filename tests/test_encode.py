from collections import Counter

import pandas as pd
import pytest

from preprocessy.encoding import EncodeData

ord_dict = {"Profession": {"Student": 1, "Teacher": 2, "HOD": 3}}
train_csv = pd.read_csv("datasets/encoding/testnew.csv")


class TestEncoding:
    # test for empty input
    def test_empty_df(self):
        params = {"target_label": "Price", "ord_dict": ord_dict}
        with pytest.raises(ValueError):
            encoder = EncodeData()
            encoder.encode(params=params)

    # test for warning
    def test_warning(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        params = {"train_df": train_csv, "ord_dict": ord_dict}
        with pytest.warns(UserWarning):
            encoder = EncodeData()
            encoder.encode(params=params)

    # test ordinal encoding
    def test_mapping(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        train_csv.drop(["Price"], axis=1, inplace=True)
        params = {
            "train_df": train_csv,
            "target_label": "Price",
            "ord_dict": ord_dict,
        }
        encoder = EncodeData()
        encoder.encode(params=params)
        assert params["train_df"]["ProfessionEncoded"].nunique() == 3
        assert params["train_df"]["ProfessionEncoded"][2] == 3
        assert Counter(params["ord_dict"]["Profession"].values()) == Counter(
            params["train_df"]["ProfessionEncoded"].unique()
        )

    # test for empty mapping
    def test_empty_weight_mapping(self):
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
            encoder = EncodeData()
            encoder.encode(params=params)

    # test for one-hot encoding
    def test_one_hot_encoding(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        params = {
            "train_df": train_csv,
            "target_label": "Price",
            "cat_cols": ["Test", "Labels"],
            "ord_dict": ord_dict,
            "one_hot": True,
        }
        encoder = EncodeData()
        encoder.encode(params=params)
        assert "Test_Tata" in params["train_df"].columns
        assert params["train_df"]["Test_Tata"][1] == 1

    def test_ignore_cat_col(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        params = {
            "train_df": train_csv,
            "target_label": "Price",
            "cat_cols": ["Profession"],
            "ord_dict": ord_dict,
            "one_hot": True,
        }
        encoder = EncodeData()
        encoder.encode(params=params)
        assert "Profession_HOD" not in params["train_df"].columns
