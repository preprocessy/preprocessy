import numpy as np
import pandas as pd
from preprocessy.feature_selection import Correlation
import pytest
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True, as_frame=True)


class TestCorrelation:
    def test_with_empty_df(self):
        with pytest.raises(ValueError):
            Correlation().find(None)

    def test_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            Correlation().find(X, 20)

    def test_with_threshold(self):
        results = Correlation().find(X, threshold=0.9)
        assert len(results) == 1

    def test_for_same_column(self):
        # A column is highly correlated with itself. Test to check if this correlation is not included.
        results = Correlation().find(X)
        for r in results:
            assert results[0] != results[1]
