import pytest

from preprocessy.pipelines.config import read_config
from preprocessy.pipelines.config import save_config


class TestConfig:
    def test_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            _ = read_config("/usr/src/app/config.json")

    def test_jsonload(self):
        with pytest.raises(TypeError):
            _ = read_config("./datasets/configs/config2.json")

    def test_df(self):
        with pytest.raises(FileNotFoundError):
            _ = read_config("./datasets/configs/config_df.json")

    def test_read(self):
        params = read_config("./datasets/configs/config4.json")
        exp = {
            "param1": 69,
            "param2": {"nestedParam": 420},
            "Split": 6969,
            "df": "./datasets/encoding/test.csv",
        }
        assert params == exp

    def test_save(self):
        filepath = "./datasets/configs/params.json"
        params = {
            "param1": 69,
            "param2": {"nestedParam": 420},
            "Split": 6969,
            "df": "./datasets/encoding/test.csv",
        }
        save_config(filepath, params)
        contents = read_config(filepath)
        params.pop("df")
        assert params == contents
