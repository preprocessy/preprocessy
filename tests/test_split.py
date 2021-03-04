import numpy as np
import pandas as pd
import pytest

from preprocessy.resampling import Split

split = Split()


class TestSplitting:
    def test_without_target_label(self):
        df = pd.DataFrame(np.arange(1000).reshape(100, 10))
        train, test = split.train_test_split(
            X=df, test_size=0.2, random_state=420
        )
        assert train.shape[0] == 80
        assert train.shape[1] == 10
        assert test.shape[0] == 20
        assert test.shape[1] == 10

    def test_random_state(self):
        df = pd.DataFrame(np.arange(1000).reshape(100, 10))
        # test passes if function raises a TypeError
        with pytest.raises(TypeError):
            train, test = split.train_test_split(
                X=df, test_size=0.2, random_state="hello"
            )
