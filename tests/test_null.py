import numpy as np
import pandas as pd
import pytest
from preprocessy.handlenullvalues import NullValuesHandler


class TestHandlingNullValues:
    def test_null_dataframe(self):
        with pytest.raises(ValueError):
            handler = NullValuesHandler()
            handler.execute({})

    def test_incorrect_input_type(self):
        self.dataframe = pd.read_csv("datasets/encoding/testnew.csv")
        with pytest.raises(TypeError):
            handler = NullValuesHandler()
            handler.execute({"df": self.dataframe, "drop": "nice"})

    def test_multiple_args(self):
        self.dataframe = pd.read_csv("datasets/encoding/testnew.csv")
        with pytest.raises(Exception):
            handler = NullValuesHandler()
            handler.execute(
                {"df": self.dataframe, "drop": True, "fill_missing": "mean"}
            )
