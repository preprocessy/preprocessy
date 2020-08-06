import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston, load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split
from preprocessy.resampling import Split

# evaluates the effect of custom train test split function on model accuracy compared to sklearn train test split
# evaluates on 4 datasets
# results stored in evaluations/results/evaluate_split.txt

splits = [None, 0.2, 0.3]


def preprocessy_eval(X, y, split, model):
    X_train, X_test, y_train, y_test = Split().train_test_split(X, y, test_size=split)
    preprocessy_test_size = None
    if split:
        preprocessy_test_size = split
    else:
        preprocessy_test_size = 1 / np.sqrt(len(X.columns))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return preprocessy_test_size, preds, y_test


def sklearn_eval(X, y, split, model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split, random_state=69
    )
    sklearn_test_size = None
    if split:
        sklearn_test_size = split
    else:
        sklearn_test_size = 0.25  # from sklearn docs
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return sklearn_test_size, preds, y_test


def eval(X, y, dataset, model):
    for split in splits:
        preprocessy_test_size, preprocessy_preds, preprocessy_y_test = preprocessy_eval(
            X, y, split, model
        )
        sklearn_test_size, sklearn_preds, sklearn_y_test = sklearn_eval(
            X, y, split, model
        )
        preprocessy_accuracy = classification_report(
            preprocessy_y_test, preprocessy_preds, output_dict=True
        )["accuracy"]
        sklearn_accuracy = classification_report(
            sklearn_y_test, sklearn_preds, output_dict=True
        )["accuracy"]
        print(f"Dataset - {dataset}")
        print(
            f"Preprocessy test_size - {preprocessy_test_size} accuracy - {preprocessy_accuracy}"
        )
        print(f"Sklearn test_size - {sklearn_test_size} accuracy - {sklearn_accuracy}")
        print()


def evaluate_on_iris():
    model = RandomForestClassifier()
    X, y = load_iris(return_X_y=True, as_frame=True)
    eval(X, y, "iris", model)


def evaluate_on_breast_cancer():
    model = RandomForestClassifier()
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    eval(X, y, "breast cancer", model)


def evaluate_on_diabetes():
    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    for split in splits:
        preprocessy_test_size, preprocessy_preds, preprocessy_y_test = preprocessy_eval(
            X, y, split, model
        )
        sklearn_test_size, sklearn_preds, sklearn_y_test = sklearn_eval(
            X, y, split, model
        )
        preprocessy_accuracy = r2_score(preprocessy_y_test, preprocessy_preds)
        sklearn_accuracy = r2_score(sklearn_y_test, sklearn_preds)
        print(f"Dataset - diabetes")
        print(
            f"Preprocessy test_size - {preprocessy_test_size} accuracy - {preprocessy_accuracy}"
        )
        print(f"Sklearn test_size - {sklearn_test_size} accuracy - {sklearn_accuracy}")
        print()


def evaluate_on_boston():
    model = LinearRegression(fit_intercept=True, normalize=True, copy_X=True)
    dataset = load_boston()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="Target")
    for split in splits:
        preprocessy_test_size, preprocessy_preds, preprocessy_y_test = preprocessy_eval(
            X, y, split, model
        )
        sklearn_test_size, sklearn_preds, sklearn_y_test = sklearn_eval(
            X, y, split, model
        )
        preprocessy_accuracy = r2_score(preprocessy_y_test, preprocessy_preds)
        sklearn_accuracy = r2_score(sklearn_y_test, sklearn_preds)
        print(f"Dataset - boston")
        print(
            f"Preprocessy test_size - {preprocessy_test_size} accuracy - {preprocessy_accuracy}"
        )
        print(f"Sklearn test_size - {sklearn_test_size} accuracy - {sklearn_accuracy}")
        print()
