import warnings

import pandas as pd

from ..exceptions import ArgumentsError


class HandleOutlier:

    """Class for handling outliers on its own or according to users needs.
    This class handles outliers using the percentile concept. The 2 percentile markers
    will represent the data to be kept ie if one marker is 5th percentile and other is 95th
    percentile then the data between this range is kept.

    Here we use the percentile to calculate the data points they point to and keep those data points
    that are in the range of the 2 calculated data points
    """

    def __init__(self):

        """Function to initialize a few parameters used to make the process
        run without human interaction.

        Parameters to be entered by the user include :
            -The dataset
            -cat_cols: Columns that are categorical should not be touched.
            -target : The target column
            -removeoutliers : default = True
            -replace : default = False => if user wants to replace
                       outliers with -999 everywhere instead of
                       removing them
            -q1 : specify the starting range (Default is 0.05)
            -q3 : specify the end of the range (Default is 0.95)

        """
        self.train_df = None
        self.test_df = None
        self.cat_cols = []
        self.ord_cols = []
        self.cols = []
        self.target_label = []
        self.remove_outliers = True
        self.replace = False
        self.quartiles = {}
        self.first_quartile = 0.05
        self.third_quartile = 0.95

    def __validate_input(self):

        if self.train_df is None:
            raise ValueError("Train dataframe should not be of None type")
        # not adding validation for whether test_df is included or not since
        # user choice

        if not isinstance(self.train_df, pd.core.frame.DataFrame):
            raise TypeError(
                "Train dataframe is not a valid dataframe.\nExpected object"
                f" type: pandas.core.frame.DataFrame\n Received type {type(self.train_df)} of dataframe"
            )
        if self.test_df is not None and not isinstance(
            self.test_df, pd.core.frame.DataFrame
        ):
            raise TypeError(
                "Test dataframe is not a valid datafram.\nExpected Object"
                f" type: pandas.core.frame.DataFrame\n Received type {type(self.test_df)} of dataframe"
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
        if not isinstance(self.cat_cols, list):
            raise TypeError(
                f"'cols' should be of type list. Received {self.cat_cols} of"
                f" type {type(self.cat_cols)}"
            )

        else:
            for c in self.cat_cols:
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

        if self.first_quartile >= 1 or self.first_quartile <= 0:
            raise ValueError(
                f"Value of first quartile must range between 0-1(exclusive).\n Rececived value {self.first_quartile}"
            )

        if (self.third_quartile >= 1) or (self.third_quartile <= 0):
            raise ValueError(
                f"Value of third quartile must range between 0-1(exclusive).\n Rececived value {self.third_quartile}"
            )

        if self.first_quartile > self.third_quartile:
            raise ValueError(
                "Value of first quartile should not be greater than value of third quartile"
            )

    def __repr__(self):
        return f"HandleOutlier(remove_outliers={self.remove_outliers}, replace={self.replace}, first_quartile={self.first_quartile}, third_quartile={self.third_quartile})"

    def __return_quartiles(self, col):
        # return the quartile range or q1 and q3 values for the column passed as parameter
        q1 = self.train_df[col].quantile(self.first_quartile)
        q1 = round(q1)
        q3 = self.train_df[col].quantile(self.third_quartile)
        q3 = round(q3)
        self.quartiles[col] = [q1, q3]

    def handle_outliers(self, params):
        """This function is used to handle outliers is flexible in how to calculate the percentiles and what to do
        about the outliers.

        :param train_df: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type train_df: pandas.core.frames.DataFrame

        :param test_df: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type test_df: pandas.core.frames.DataFrame

        :param target_label: Name of the Target Column. This parameter is needed to ensure that the target column isn't included in the outlier removing process.
        :type target_label: str

        :param cat_cols: List containing the column names to be encoded categorically. This parameter is needed to ensure that the categorical columns isn't included in the outlier removing process.
        :type cat_cols: List

        :param remove_outliers: Boolean value to indicate whether user wants to remove outlier or not.
        :type remove_outliers: bool, default=True

        :param ord_cols: List containing the column names to be encoded ordinally.This parameter is needed to ensure that the ordinal columns isn't included in the outlier removing process.
        :type ord_cols: List

        :param replace: Integer value to indicate the value with which to replace the identified outliers.
                        This will replace and will not remove the outliers
        :type replace: int

        :param first_quartile: Float value <1 representing the first percentile marker
        :type first_quartile: float

        :param third_quartile: Float value <1 representing the other percentile marker.
        :type third_quartile: float

        """
        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "target_label" in params.keys():
            self.target_label.append(params["target_label"])
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "ord_cols" in params.keys():
            self.ord_cols = params["ord_cols"]
        if "remove_outliers" in params.keys():
            self.remove_outliers = params["remove_outliers"]
        if "replace" in params.keys():
            self.replace = params["replace"]
        if "first_quartile" in params.keys():
            self.first_quartile = params["first_quartile"]
        if "third_quartile" in params.keys():
            self.third_quartile = params["third_quartile"]

        self.__validate_input()

        if "out_cols" in params.keys():
            self.cols = params["out_cols"]
            self.cols = [
                item
                for item in self.cols
                if item not in self.cat_cols and item not in self.target_label
            ]

        else:
            self.cols = []
            for col in self.train_df.columns:
                if (
                    col not in self.cat_cols
                    and col not in self.ord_cols
                    and col not in self.target_label
                    and not isinstance(self.train_df.iloc[0][col], str)
                ):
                    self.cols.append(col)

        # if user has marked removeoutliers = True and wants outliers removed..
        if self.remove_outliers:
            if len(self.cols) >= 1:
                for col in self.cols:
                    self.__return_quartiles(col)
                for col in self.cols:
                    q1, q3 = self.quartiles[col]
                    self.train_df = self.train_df[(self.train_df[col] > q1)]
                    self.train_df = self.train_df[(self.train_df[col] <= q3)]
                    if self.test_df is not None:
                        self.test_df = self.test_df[(self.test_df[col] > q1)]
                        self.test_df = self.test_df[(self.test_df[col] <= q3)]
        # if removeoutliers = False and replace=True i.e. user wants outliers
        # replaced by a value to indicate these are outliers
        elif self.replace:
            if len(self.cols) >= 1:
                for col in self.cols:
                    self.__return_quartiles(col)
                for col in self.cols:
                    q1, q3 = self.quartiles[col]
                    self.train_df[(self.train_df[col] < q1)] = -999
                    self.train_df[(self.train_df[col] > q3)] = -999
                    if self.test_df is not None:
                        self.test_df[(self.test_df[col] <= q1)] = -999
                        self.test_df[(self.test_df[col] >= q3)] = -999

        params["train_df"] = self.train_df
        params["test_df"] = self.test_df
