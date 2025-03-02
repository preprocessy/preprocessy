import numpy as np
import pandas as pd
import pytest
from preprocessy.data_splitting import Split


def test_without_target_col():
    df = pd.DataFrame(np.arange(1000).reshape(100, 10))
    params = {
        "train_df": df,
        "test_size": 0.2,
        "random_state": 420,
        "shuffle": True,
    }
    split = Split()
    split.train_test_split(params=params)
    assert params["X_train"].shape[0] == 80
    assert params["X_train"].shape[1] == 10
    assert params["X_test"].shape[0] == 20
    assert params["X_test"].shape[1] == 10


def test_random_state():
    df = pd.DataFrame(np.arange(1000).reshape(100, 10))
    params = {"train_df": df, "test_size": 0.2, "random_state": "hello"}
    split = Split()
    with pytest.raises(TypeError):
        split.train_test_split(params=params)
