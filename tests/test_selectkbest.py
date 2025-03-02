import pandas as pd
import pytest
from preprocessy.feature_selection import SelectKBest
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression

X_reg, y_reg = make_regression(n_samples=1000, n_features=50)
X_reg = pd.DataFrame(X_reg)
X_reg["target"] = pd.Series(y_reg)

X_class, y_class = make_classification(n_samples=1000, n_features=50)
X_class = pd.DataFrame(X_class)
X_class["target"] = pd.Series(y_class)


def test_invalid_input():
    kbest = SelectKBest()
    with pytest.raises(TypeError):
        kbest.fit_transform({})


@pytest.mark.parametrize(
    "test_input",
    [
        {
            "score_func": 10,
            "k": 10,
            "train_df": X_reg,
            "target_label": "target",
        },
        {
            "score_func": "chi2",
            "k": 10,
            "train_df": X_reg,
            "target_label": "target",
        },
    ],
)
def test_score_func(test_input):
    kbest = SelectKBest()
    with pytest.raises(TypeError):
        kbest.fit(params=test_input)


@pytest.mark.parametrize(
    "test_input, test_output",
    [
        (
            {"k": 10, "train_df": X_reg, "target_label": "target"},
            f_regression.__name__,
        ),
        ({"train_df": X_class, "target_label": "target"}, f_classif.__name__),
    ],
)
def test_default_score_func(test_input, test_output):
    kbest = SelectKBest()
    kbest.fit(params=test_input)
    assert kbest.score_func.__name__ == test_output


def test_transform_without_fit():
    kbest = SelectKBest()
    with pytest.raises(ValueError):
        kbest.transform({"train_df": X_reg, "target_label": "target"})


@pytest.mark.parametrize(
    "params, split_size",
    [
        ({"train_df": X_reg, "target_label": "target", "k": 10}, 1000),
        ({"train_df": X_class, "target_label": "target", "k": 10}, 1000),
        (
            {
                "train_df": X_reg.iloc[:800, :],
                "test_df": X_reg.iloc[800:, :],
                "target_label": "target",
                "k": 10,
                "score_func": f_regression,
            },
            800,
        ),
        (
            {
                "train_df": X_class.iloc[:800, :],
                "test_df": X_class.iloc[800:, :],
                "target_label": "target",
                "k": 10,
                "score_func": f_classif,
            },
            800,
        ),
    ],
)
def test_fit_transform(params, split_size):
    kbest = SelectKBest()
    kbest.fit_transform(params)
    assert params["train_df"].shape == (split_size, params["k"] + 1)
    if "test_df" in params.keys():
        assert params["test_df"].shape == (1000 - split_size, params["k"] + 1)
