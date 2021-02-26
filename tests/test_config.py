import pytest
from preprocessy.pipelines import Config

class TestConfig:

    def test_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            c = Config("/usr/src/app/config.json",["Split"])
    
    def test_jsonload(self):
        with pytest.raises(TypeError):
            c = Config("./datasets/configs/config2.json",["Split"])

    def test_step(self):
        with pytest.raises(ValueError):
            c = Config("./datasets/configs/config3.json",["Split"])
            c.readConfig()
        
    def test_read(self):
        c = Config("./datasets/configs/config4.json",["Split"])
        params = c.readConfig()
        exp = {"param1":69,"param2":{"nestedParam":420},"Split":6969}
        assert (
            params == exp
        )