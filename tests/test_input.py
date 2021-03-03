from preprocessy.input import ReadData
import pandas as pd
import numpy as np
import pytest

reader = ReadData()


class TestReader:
    def test_incorrect_file_name_type(self):
        with pytest.raises(TypeError):
            reader.read_file({"df_path": None})

        with pytest.raises(TypeError):
            reader.read_file({"df_path": ["datasets/encoding/test.csv"]})

    def test_incorrect_file_type(self):
        with pytest.raises(ValueError):
            reader.read_file({"df_path": "datasets"})

    def test_reader(self):
        reader.read_file({"df_path": "datasets/encoding/test.csv"})
        assert isinstance(reader.df, pd.core.frame.DataFrame) == True
        assert reader.df.shape == (3, 4)
        assert reader.stats.iloc[1, 0] == -4003

    def test_file_not_exists(self):
        with pytest.raises(FileNotFoundError):
            reader.read_file({"df_path": "hello.csv"})
