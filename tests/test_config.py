import pytest

from preprocessy.pipelines.config import read_config
from preprocessy.pipelines.config import save_config


def test_filenotfound():
    with pytest.raises(FileNotFoundError):
        _ = read_config("/usr/src/app/config.json")


def test_jsonload():
    with pytest.raises(TypeError):
        _ = read_config("./datasets/configs/config2.json")


def test_df():
    with pytest.raises(FileNotFoundError):
        _ = read_config("./datasets/configs/config_df.json")


def test_read():
    params = read_config("./datasets/configs/config4.json")
    exp = {
        "param1": 69,
        "param2": {"nestedParam": 420},
        "Split": 6969,
        "df": "./datasets/encoding/test.csv",
    }
    assert params == exp


def test_save():
    filepath = "./datasets/configs/params.json"
    params = {
        "param1": 69,
        "param2": {"nestedParam": 420},
        "Split": 6969,
    }
    save_config(filepath, params)
    contents = read_config(filepath)
    assert params == contents
