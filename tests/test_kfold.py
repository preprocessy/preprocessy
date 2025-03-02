import numpy as np
import pandas as pd
import pytest
from preprocessy.data_splitting import KFold


@pytest.mark.parametrize(
    "n_splits, shuffle, random_state",
    [
        (2.3, False, None),
        (0, False, None),
        (5, 1, None),
        (5, True, 4.5),
        (5, False, 69),
    ],
)
def test_kfold(n_splits, shuffle, random_state):
    df = pd.DataFrame(np.arange(1000).reshape(100, 10))
    cv = KFold()
    params = {
        "train_df": df,
        "n_splits": n_splits,
        "shuffle": shuffle,
        "random_state": random_state,
    }
    with pytest.raises(ValueError):
        for _, _ in cv.split(params):
            pass


def test_split():
    df = pd.DataFrame(np.arange(1000).reshape(100, 10))
    cv = KFold()
    params = {"train_df": df, "n_splits": 5, "shuffle": True}
    for train_indices, test_indices in cv.split(params):
        assert len(train_indices) == 80
        assert len(test_indices) == 20
