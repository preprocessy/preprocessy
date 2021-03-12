import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression

from preprocessy.feature_selection import SelectKBest

X_reg, y_reg = make_regression(n_samples=1000, n_features=50)
X_reg = pd.DataFrame(X_reg)
y_reg = pd.Series(y_reg)
X_class, y_class = make_classification(n_samples=1000, n_features=50)
X_class = pd.DataFrame(X_class)
y_class = pd.Series(y_class)


def test_invalid_input():
    kbest = SelectKBest()
    with pytest.raises(TypeError):
        kbest.fit_transform({})


@pytest.mark.parametrize(
    "test_input",
    [
        {"score_func": 10, "k": 10, "X": X_class, "y": y_reg},
        {"score_func": "chi2", "k": 10, "X": X_class, "y": y_reg},
    ],
)
def test_score_func(test_input):
    with pytest.raises(TypeError):
        kbest = SelectKBest()
        kbest.fit(params=test_input)


@pytest.mark.parametrize(
    "test_input, test_output",
    [
        ({"k": 10, "X": X_reg, "y": y_reg}, f_regression.__name__),
        ({"X": X_class, "y": y_class}, f_classif.__name__),
    ],
)
def test_default_score_func(test_input, test_output):
    kbest = SelectKBest()
    kbest.fit(params=test_input)
    assert kbest.score_func.__name__ == test_output


def test_transform_without_fit():
    kbest = SelectKBest()
    with pytest.raises(ValueError):
        kbest.transform(X_reg)


def test_fit_transform():
    kbest = SelectKBest()
    params = {"score_func": f_regression, "k": 5, "X": X_reg, "y": y_reg}
    kbest.fit_transform(params)
    assert params["X_best"].shape[1] == 5
