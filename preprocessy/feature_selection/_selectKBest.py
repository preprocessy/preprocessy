import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression


class SelectKBest:
    """Class for finding K highest scoring features among the set of all features. Takes a feature and finds its correlation with the
    target label using a scoring function.

    Scoring functions include:

    1. f_regression
    2. mutual_info_regression
    3. f_classif
    4. mutual_info_classif
    5. chi2

    All scoring functions are provided in sklearn.feature_selection

    Private Methods
    ---------------

    __get_mask() : Generates mask that contains features to be selected

    Public Methods
    --------------

    fit() : Generates scores and pvalues by fitting the scoring function over the dataset

    transform() : Uses the mask to select the k best features

    fit_transform() : Performs fit() then transform()

    """

    def __init__(self):
        """Constructor

        Parameters
        ----------

        train_df : pd.core.frame.DataFrame, default=None. This is the training dataset
                    consisting of features as well as the target column

        test_df : pd.core.frame.DataFrame, default=None. This is the testing dataset
                    consisting of features as well as the target column

        target_col : str, default=None. The name of the target column

        score_func : callable, default=None. Function taking two arrays X and y, and returning a pair of arrays
                     (scores, pvalues) or a single array with scores. score_func is provided from sklearn.feature_selection

        k : int, default=10. Number of top features to select.

        """
        self.score_func = None
        self.k = 10
        self.scores = None
        self.pvalues = None
        self.train_df = None
        self.test_df = None
        self.target_col = None
        self.X = None
        self.y = None

    def __repr__(self):
        return f"SelectKBest(score_func={self.score_func}, k={self.k})"

    def __validate_input(self):
        if not isinstance(self.train_df, pd.core.frame.DataFrame):
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )
        if self.test_df is not None:
            if not isinstance(self.test_df, pd.core.frame.DataFrame):
                raise TypeError(
                    "Test dataframe is not a valid dataframe.\nExpected object"
                    " type: pandas.core.frame.DataFrame"
                )
        if not isinstance(self.target_col, str):
            raise TypeError(
                f"Target Column name should be of type 'str'. Received {self.target_col} of type {type(self.target_col)}"
            )

        if self.target_col not in self.train_df.columns:
            raise ValueError(
                f"Target column {self.target_col} not found in train_df"
            )

        if self.test_df is not None:
            if self.target_col not in self.test_df.columns:
                raise ValueError(
                    f"Target column {self.target_col} not found in test_df"
                )

    def __get_mask(self):
        """Function to generate mask for selecting features. Uses the scores generated by the scoring function to select the k
        highest scoring features.

        Returns
        -------

        mask : ndarray of type=bool, shape=(n_features,). Boolean mask of features to be selected.

        """
        if self.k == 0:
            return np.zeros(self.scores.shape, dtype=bool)

        if self.scores is None:
            raise ValueError(
                "self.scores is None. Please fit the estimator before calling"
                " transform."
            )

        mask = np.zeros(self.scores.shape, dtype=bool)
        # select k highest scored features
        mask[np.argsort(self.scores, kind="stable")[-self.k :]] = 1
        return mask

    def fit(self, params):
        """Function that fits the scoring function over (X_train, y_train) and generates the scores and pvalues for all features with the
        target label. If no scoring function is passed, then defaults to f_classify or f_regression based on the predictive
        problem.

        Parameters
        ----------

        X : pd.core.frame.DataFrame of shape (n_samples, n_features). The training input features.

        y : pd.core.series.Series of shape (n_samples,). The target values (class labels in classification, real numbers in
            regression).

        """

        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "target_col" in params.keys():
            self.target_col = params["target_col"]
        if "score_func" in params.keys():
            self.score_func = params["score_func"]
        if "k" in params.keys():
            self.k = params["k"]

        self.__validate_input()

        self.X = self.train_df.drop(self.target_col, axis=1)
        self.y = self.train_df[self.target_col]

        if self.score_func is None:
            if self.y.nunique() <= 15:
                self.score_func = f_classif
            else:
                self.score_func = f_regression

        if not callable(self.score_func):
            raise TypeError(
                f"The score function should be a callable, {self.score_func}"
                f" of type ({type(self.score_func)}) was passed."
            )

        score_func_ret = self.score_func(self.X, self.y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores, self.pvalues = score_func_ret
            self.pvalues = np.asarray(self.pvalues)
        else:
            self.scores = score_func_ret
            self.pvalues = None

        self.scores = np.asarray(self.scores)
        params["score_func"] = self.score_func

    def transform(self, params):
        """Function to reduce X_train and X_test to the selected features. Returns dataframes of shape (n_samples, k)

        Parameters
        ----------

        X : pd.core.frame.DataFrame of shape (n_samples, n_features). The input samples.

        Returns
        -------

        X_train : array of shape (n_samples, k). The input samples with only the selected features.

        X_test : array of shape (n_test_samples, k). The test samples with only the selected features.

        """
        if "score_func" in params.keys():
            self.score_func = params["score_func"]
        if "k" in params.keys():
            self.k = params["k"]

        mask = self.__get_mask()
        if not mask.any():
            raise ValueError(
                "No features were selected: either the data is too noisy or"
                " the selection test too strict."
            )

        if len(mask) != self.X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        self.X = self.X.iloc[:, mask]
        self.X[self.target_col] = self.y
        params["train_df"] = self.X
        if self.test_df is not None:
            self.X = self.test_df.drop(self.target_col, axis=1).iloc[:, mask]
            self.X[self.target_col] = self.test_df[self.target_col]
            params["test_df"] = self.X

    def fit_transform(self, params):
        """Does fit() and transform() in single step

        Parameters
        ----------

        train_df : pd.core.frame.DataFrame, default=None. This is the training dataset
                    consisting of features as well as the target column

        test_df : pd.core.frame.DataFrame, default=None. This is the testing dataset
                    consisting of features as well as the target column

        target_col : str, default=None. The name of the target column

        score_func : callable, default=None. Function taking two arrays X and y, and returning a pair of arrays
                     (scores, pvalues) or a single array with scores. score_func is provided from sklearn.feature_selection

        k : int, default=10. Number of top features to select.

        Returns
        -------

        X_train : array of shape (n_samples, k). The input samples with only the selected features.

        X_test : array of shape (n_test_samples, k). The test samples with only the selected features.

        """

        self.fit(params)
        self.transform(params)
