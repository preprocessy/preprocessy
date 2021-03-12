import pytest
from sklearn.datasets import load_iris

from preprocessy.feature_selection import Correlation

X, y = load_iris(return_X_y=True, as_frame=True)


def test_with_empty_df():
    with pytest.raises(ValueError):
        for tup in Correlation().find(None):
            print(tup)


def test_with_invalid_threshold():
    with pytest.raises(ValueError):
        for tup in Correlation().find(X, 20):
            print(tup)


def test_with_threshold():
    count = 0
    for _ in Correlation().find(X, threshold=0.9):
        count += 1
    assert count == 1


def test_for_same_column():
    # A column is highly correlated with itself. Test to check that this correlation is not included.
    for tup in Correlation().find(X):
        assert tup[0] != tup[1]
