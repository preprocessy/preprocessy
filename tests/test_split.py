import numpy as np
import pandas as pd
import pytest

from preprocessy.resampling import Split



def test_without_target_col():
    df = pd.DataFrame(np.arange(1000).reshape(100, 10))
    params = {"X": df, "test_size": 0.2, "random_state": 420}
    split = Split()
    split.train_test_split(params=params)
    assert params["train"].shape[0] == 80
    assert params["train"].shape[1] == 10
    assert params["test"].shape[0] == 20
    assert params["test"].shape[1] == 10

def test_random_state():
    df = pd.DataFrame(np.arange(1000).reshape(100, 10))
    # test passes if function raises a TypeError
    with pytest.raises(TypeError):
        params = {"X": df, "test_size": 0.2, "random_state": "hello"}
        split = Split()
        split.train_test_split(params=params)

def test_without_target_label():
    df_x = pd.DataFrame(np.arange(1000).reshape(100, 10))
    df_y = pd.Series(np.arange(100))
    with pytest.raises(ValueError):
        params = {
            "X": df_x,
            "y": df_y,
            "test_size": 0.2,
            "random_state": 420,
        }
        split = Split()
        split.train_test_split(params=params)