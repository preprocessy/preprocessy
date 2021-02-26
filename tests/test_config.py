from tests.test_base import read
import pytest
from preprocessy.pipelines.config import read_config, save_config

class TestConfig:

    def test_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            c = read_config("/usr/src/app/config.json",["Split"])
    
    def test_jsonload(self):
        with pytest.raises(TypeError):
            c = read_config("./datasets/configs/config2.json",["Split"])

    def test_step(self):
        with pytest.raises(ValueError):
            c = read_config("./datasets/configs/config3.json",["Split"])
        
    def test_read(self):
        params = read_config("./datasets/configs/config4.json",["Split"])
        exp = {"param1":69,"param2":{"nestedParam":420},"Split":6969}
        assert (
            params == exp
        )

    def test_save(self):
        filepath = "./datasets/configs/params.json"
        params = {"param1":69,"param2":{"nestedParam":420},"Split":6969}
        save_config(filepath,params)
        contents = read_config(filepath,["Split"])
        assert (
            params == contents
        )
