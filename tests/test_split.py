import numpy as np
import pandas as pd
import pytest

from preprocessy.resampling import Split

split = Split()


class TestSplitting:
    def test_without_target_label(self):
        df = pd.DataFrame(np.arange(1000).reshape(100, 10))
        params = {"X": df, "test_size": 0.2, "random_state": 420}
        split.train_test_split(params=params)
        assert params["train"].shape[0] == 80
        assert params["train"].shape[1] == 10
        assert params["test"].shape[0] == 20
        assert params["test"].shape[1] == 10

    def test_random_state(self):
        df = pd.DataFrame(np.arange(1000).reshape(100, 10))
        # test passes if function raises a TypeError
        with pytest.raises(TypeError):
            params = {"X": df, "test_size": 0.2, "random_state": "hello"}
            split.train_test_split(params=params)
