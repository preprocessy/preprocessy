from preprocessy.encoding import EncodeData
import pandas as pd
import pytest
from collections import Counter

ord_dict = {"Profession": {"Student": 1, "Teacher": 2, "HOD": 3}}

params = {"cat_cols": [], "ord_dict": ord_dict}


class TestOrdinalEncoding:
    # test for empty input
    def test_empty_df(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        with pytest.raises(ValueError):
            encoder = EncodeData(target_label="Price", params=params)
            train = encoder.encode()

    # test for warning
    def test_warning(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        with pytest.warns(UserWarning):
            encoder = EncodeData(train_df=train_csv, params=params)
            train = encoder.encode()

    # test ordinal encoding
    def test_mapping(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        encoder = EncodeData(train_df=train_csv, target_label="Price", params=params)
        train = encoder.encode()
        assert train["Profession"].nunique() == 3
        assert train["Profession"][2] == 3
        assert Counter(params["ord_dict"]["Profession"].values()) == Counter(
            train["Profession"].unique()
        )

    # test for empty mapping
    def test_empty_weight_mapping(self):
        train_csv = pd.read_csv("datasets/encoding/testnew.csv")
        with pytest.raises(ValueError):
            params["ord_dict"]["Size"] = None
            encoder = EncodeData(
                train_df=train_csv, target_label="Price", params=params
            )
            train = encoder.encode()
