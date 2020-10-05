import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.feature_selection import f_classif, f_regression
import pytest

from preprocessy.feature_selection import SelectKBest

X_reg, y_reg = make_regression(n_samples=1000, n_features=50)
X_reg = pd.DataFrame(X_reg)
y_reg = pd.Series(y_reg)
X_class, y_class = make_classification(n_samples=1000, n_features=50)
X_class = pd.DataFrame(X_class)
y_class = pd.Series(y_class)


class TestSelectKBest:
    def test_invalid_input(self):
        kbest = SelectKBest()
        with pytest.raises(TypeError):
            kbest.fit_transform(None)

    def test_score_func(self):
        with pytest.raises(TypeError):
            kbest = SelectKBest(score_func=10, k=10)
            kbest.fit(X_class, y_reg)
        with pytest.raises(TypeError):
            kbest = SelectKBest(score_func="chi2", k=10)
            kbest.fit(X_class, y_reg)

    def test_default_score_func(self):
        kbest = SelectKBest(k=10)
        kbest.fit(X_reg, y_reg)
        assert kbest.score_func.__name__ == f_regression.__name__

        kbest = SelectKBest()
        kbest.fit(X_class, y_class)
        assert kbest.score_func.__name__ == f_classif.__name__

    def test_transform_without_fit(self):
        kbest = SelectKBest()
        with pytest.raises(ValueError):
            kbest.transform(X_reg)

    def test_fit_transform(self):
        kbest = SelectKBest(score_func=f_regression, k=5)
        X_new = kbest.fit_transform(X_reg, y_reg)
        assert X_new.shape[1] == 5
