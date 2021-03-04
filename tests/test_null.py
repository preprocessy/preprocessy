import numpy as np
import pandas as pd
import pytest
from preprocessy.handlenullvalues import NullValuesHandler
from preprocessy.handlenullvalues.errors import ArgumentsError

dataframe = pd.read_csv("datasets/encoding/test2.csv")


class TestHandlingNullValues:
    def test_null_dataframe(self):
        with pytest.raises(ValueError):
            handler = NullValuesHandler()
            handler.execute({})

    def test_none_args(self):
        with pytest.raises(ArgumentsError):
            handler = NullValuesHandler()
            handler.execute({"df":dataframe})

    def test_multiple_args(self):
        with pytest.raises(ArgumentsError):
            handler = NullValuesHandler()
            handler.execute({"df": dataframe, "drop": True, "fill_missing":"mean"})

        with pytest.raises(ArgumentsError):
            handler = NullValuesHandler()
            handler.execute({"df": dataframe, "drop": True, "fill_values":{"Test": "Tata"}})

        with pytest.raises(ArgumentsError):
            handler = NullValuesHandler()
            handler.execute({"df":dataframe, "drop":True, "fill_values":{"Test": "Tata"}})

        with pytest.raises(ArgumentsError):
            handler = NullValuesHandler(
            )
            handler.execute({"df":dataframe, "fill_missing":"mean", "fill_values":{"Test": "Tata"}})

    def test_incorrect_input_type(self):
        # for dataframe argument
        array = np.random.random((5, 5))
        with pytest.raises(TypeError):
            handler = NullValuesHandler()
            handler.execute({"df":array})

        # for drop argument
        self.dataframe = pd.read_csv("datasets/encoding/testnew.csv")

        with pytest.raises(TypeError):
            handler = NullValuesHandler()
            handler.execute({"df": self.dataframe, "drop": "nice"})

        # for fill_missing argument
        with pytest.raises(TypeError):
            handler = NullValuesHandler()
            handler.execute({"df":dataframe, "fill_missing":3})

        with pytest.raises(ArgumentsError):
            handler = NullValuesHandler()
            handler.execute({"df":dataframe, "fill_missing":"sum"})

        # for fill_values argument
        with pytest.raises(TypeError):
            value = [5]
            handler = NullValuesHandler()
            handler.execute({"df":dataframe, "fill_values":value})

        with pytest.raises(ArgumentsError):
            value = {"Label": 10}
            handler = NullValuesHandler()
            handler.execute({"df":dataframe, "fill_values":value})

    # to test output
    def test_output(self):
        params = {"df":dataframe, "drop":True}
        handler = NullValuesHandler()
        handler.execute(params=params)
        assert params["df"].isnull().values.any() == False

        params = {"df":dataframe, "fill_missing":"mean"}
        handler = NullValuesHandler()
        handler.execute(params=params)
        assert params["df"].isnull().any()["Distance"] == False

        params = {"df":dataframe, "fill_values":{"Distance": 0}}
        handler = NullValuesHandler()
        handler.execute(params=params)
        assert params["df"].isnull().any()["Distance"] == False

    # for dropping list of columns passed by user
    def test_drop_col(self):
        params = {"df":dataframe, "drop":True, "column_list":["Distance"]}
        handler = NullValuesHandler()
        handler.execute(params=params)
        assert "Distance" not in params["df"].columns
