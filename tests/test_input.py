import pandas as pd
import pytest

from preprocessy.input import ReadData

reader = ReadData()


@pytest.mark.parametrize(
    "test_input",
    [{"df_path": None}, {"df_path": ["datasets/encoding/test.csv"]}],
)
def test_incorrect_file_name_type(test_input):
    with pytest.raises(TypeError):
        reader.read_file(params=test_input)


def test_incorrect_file_type():
    with pytest.raises(ValueError):
        reader.read_file({"df_path": "datasets"})


def test_reader():
    reader.read_file({"df_path": "datasets/encoding/test.csv"})
    assert isinstance(reader.df, pd.core.frame.DataFrame)
    assert reader.df.shape == (3, 5)
    assert reader.stats.iloc[1, 0] == -4003


def test_file_not_exists():
    with pytest.raises(FileNotFoundError):
        reader.read_file({"df_path": "hello.csv"})
