import warnings

import pandas as pd

from ..exceptions import ArgumentsError


class HandleOutlier:

    """Class for handling outliers on its own or according to users needs.

     Private methods
    _ _ _ _ _ _ _ _ _ _

    __return_quartiles() : returns the 5% and 95% mark in the distribution
                          of data(Values above are default values)

     Public Methods
    _ _ _ _ _ _ _ _ _

    handle_outliers() : Takes in the dataset as input, finds the quartiles
                       and returns the dataset within the interquartile
                       range. Function run only on int64 and float64
                       specified columns.

    """

    def __init__(self):

        """Function to initialize a few parameters used to make the process
        run without human interaction.

        Parameters to be entered by the user include :
            -The dataset
            -cols: Any specific columns in dataset the user wants to
                   remove outliers from. If not entered columns will be
                   selected based on their dtype(int and float only) and
                   outliers will be removed from them
            -removeoutliers : default = True
            -replace : default = False => if user wants to replace
                       outliers with -999 everywhere instead of
                       removing them
            -q1 : specify the starting range (Default is 0.05)
            -q3 : specify the end of the range (Default is 0.95)

        """
        self.train_df = None
        self.cols = []
        self.remove_outliers = True
        self.replace = False
        self.quartiles = {}
        self.first_quartile = 0.05
        self.third_quartile = 0.95

    def __validate_input(self):

        if self.train_df is None:
            raise ValueError("Train dataframe should not be of None type")

        if not isinstance(self.train_df, pd.core.frame.DataFrame):
            raise TypeError(
                "Train dataframe is not a valid dataframe.\nExpected object"
                f" type: pandas.core.frame.DataFrame\n Received type {type(self.train_df)} of dataframe"
            )

        if not isinstance(self.cols, list):
            raise TypeError(
                f"'cols' should be of type list. Received {self.cols} of"
                f" type {type(self.cols)}"
            )

        else:
            for c in self.cols:
                if not isinstance(c, str):
                    raise TypeError(
                        f"'column' should be of type str. Received {c} of"
                        f" type {type(c)}"
                    )
                elif c not in self.train_df.columns:
                    raise KeyError(f" '{c}' column is not present in train_df")

        if not isinstance(self.remove_outliers, bool):
            raise TypeError(
                f"'remove_outliers' should be of type bool. Received {self.remove_outliers} of"
                f" type {type(self.remove_outliers)}"
            )

        if not isinstance(self.replace, bool):
            raise TypeError(
                f"'replace' should be of type bool. Received {self.replace} of"
                f" type {type(self.replace)}"
            )

        if self.remove_outliers and self.replace:
            raise ArgumentsError(
                "Both remove_outliers and replace arguments cannot be true"
            )

        if (not self.remove_outliers) and (not self.replace):
            warnings.warn(
                "remove_outliers and replace both are False, thus no operation will be performed on"
                " dataframe, please specify either of the argument as True ",
                UserWarning,
            )

        if not isinstance(self.first_quartile, float):
            raise TypeError(
                f"'first_quartile' should be of type float. Received {self.first_quartile} of"
                f" type {type(self.first_quartile)}"
            )

        if not isinstance(self.third_quartile, float):
            raise TypeError(
                f"'third_quartile' should be of type float. Received {self.third_quartile} of"
                f" type {type(self.third_quartile)}"
            )

    def __repr__(self):
        return f"HandleOutlier(remove_outliers={self.remove_outliers}, replace={self.replace}, first_quartile={self.first_quartile}, third_quartile={self.third_quartile})"

    def __return_quartiles(self, col):
        # return the quartile range or q1 and q3 values for the column passed as parameter
        train_df = self.train_df
        q1 = train_df[col].quantile(self.first_quartile)
        q1 = round(q1)
        q3 = train_df[col].quantile(self.third_quartile)
        q3 = round(q3)
        self.quartiles[col] = [q1, q3]

    def handle_outliers(self, params):

        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "cols" in params.keys():
            self.cols = params["cols"]
        if "remove_outliers" in params.keys():
            self.remove_outliers = params["remove_outliers"]
        if "replace" in params.keys():
            self.replace = params["replace"]
        if "first_quartile" in params.keys():
            self.first_quartile = params["first_quartile"]
        if "third_quartile" in params.keys():
            self.third_quartile = params["third_quartile"]

        self.__validate_input()

        # parameters till now: train_df, cols, removeoutliers, replace
        # if user has marked removeoutliers = True and wants outliers removed..
        train_df = self.train_df
        if self.remove_outliers:
            if len(self.cols) >= 1:
                for col in self.cols:
                    self.__return_quartiles(col)
                for col in self.cols:
                    q = self.quartiles[col]
                    q1 = q[0]
                    q3 = q[1]
                    train_df = train_df[(train_df[col] >= q1)]
                    train_df = train_df[(train_df[col] <= q3)]

        # if removeoutliers = False and replace=True i.e. user wants outliers
        # replaced by a value to indicate these are outliers
        elif self.replace:
            if len(self.cols) >= 1:
                for col in self.cols:
                    self.__return_quartiles(col)
                for col in self.cols:
                    q = self.quartiles[col]
                    q1 = q[0]
                    q3 = q[1]
                    train_df[(train_df[col] < q1)] = -999
                    train_df[(train_df[col] > q3)] = -999
        self.train_df = train_df
        params["train_df"] = self.train_df
