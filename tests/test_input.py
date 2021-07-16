import pandas as pd
import pytest
from preprocessy.input import Reader

reader = Reader()


@pytest.mark.parametrize(
    "test_input",
    [
        {"train_df_path": None},
        {"train_df_path": None, "test_df_path": None},
        {"train_df_path": ["datasets/encoding/test.csv"]},
    ],
)
def test_incorrect_file_name_type(test_input):
    with pytest.raises(TypeError):
        reader.read_file(params=test_input)


def test_incorrect_file_type():
    with pytest.raises(ValueError):
        reader.read_file({"train_df_path": "datasets"})


@pytest.mark.parametrize(
    "test_input",
    [
        {"train_df_path": "datasets/encoding/test.csv"},
        {
            "train_df_path": "datasets/encoding/test.csv",
            "test_df_path": "datasets/encoding/test.csv",
        },
    ],
)
def test_reader(test_input):
    reader.read_file(test_input)
    assert isinstance(test_input["train_df"], pd.core.frame.DataFrame)
    assert test_input["train_df"].shape == (3, 5)


def test_file_not_exists():
    with pytest.raises(FileNotFoundError):
        reader.read_file({"train_df_path": "hello.csv"})
