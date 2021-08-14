import warnings

import numpy as np
import pandas as pd

from ..exceptions import ArgumentsError


class NullValuesHandler:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.drop_cols = None
        self.cat_cols = []
        self.replace_cat_nulls = None
        self.fill_missing = None
        self.fill_values = None
        self.new_df = None
        self.final_train = None
        self.final_test = None
        self.final_df = None

    def __repr__(self):
        return f"NullValuesHandler(train_df=None, test_df=None, drop_cols={self.drop_cols}, fill_missing={self.fill_missing}, fill_values={self.fill_values})"

    def __validate_input(self):

        if self.train_df is None:
            raise ValueError("Feature dataframe should not be of None type")

        if type(self.train_df) is not pd.core.frame.DataFrame:
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )
        col_list = list(self.train_df)

        if self.test_df is not None:
            if type(self.test_df) is not pd.core.frame.DataFrame:
                raise TypeError(
                    "Feature dataframe is not a valid dataframe.\nExpected object"
                    " type: pandas.core.frame.DataFrame"
                )
            test_col_list = list(self.test_df)
            if test_col_list != col_list:
                raise ArgumentsError(
                    "The columns in train df and test df are not same. Provide test and train df with same columns."
                )
        # if (
        #     self.drop_cols is None
        #     and self.fill_missing is None
        #     and self.fill_values is None
        # ):
        #     raise ArgumentsError(
        #         "Expected atleast one argument apart from dataframe, received none"
        #     )

        if self.drop_cols is not None:
            if not isinstance(self.drop_cols, list):
                raise TypeError('Expected List for argument "drop_cols"')
            if len(self.drop_cols) == 0:
                warnings.warn(
                    '"drop_cols" is empty, no columns will be dropped.',
                    UserWarning,
                )
            if len(self.drop_cols) != 0:
                for c in self.drop_cols:
                    if c not in col_list:
                        raise ArgumentsError(
                            f'Column "{c}" does not exist in dataframe'
                        )

        if self.fill_missing is not None:
            dtypeList = [np.int64, np.int32, np.float32, np.float64]
            if type(self.fill_missing) is not list:
                raise TypeError(
                    'Expected list for argument "fill_missing" '
                )
            self.fill_missing[1] = self.fill_missing[1].lower()
            if self.fill_missing[1] not in ["mean", "median"]:
                raise ArgumentsError('Allowed argument is "mean" or "median" ')
            
            if type(self.fill_missing[0]) is list and len(self.fill_missing[0]) == 0:
                warnings.warn(
                    f'No columns specified."{self.fill_missing[1]}" will be applied on all columns containing null values.',
                    UserWarning,
                )
            if len(self.fill_missing[0]) != 0:
                for c in self.fill_missing[0]:
                    if c not in col_list:
                        raise ArgumentsError(
                            f'Column "{c}" does not exist in dataframe'
                        )
                    else:
                        if self.train_df.dtypes[c] not in dtypeList:
                            raise TypeError(
                                f'Expected integer or float datatype in columns to be filled with mean or median. Column in error here : "{c}"'
                            )

        if self.fill_values is not None:
            if type(self.fill_values) is not dict:
                raise TypeError(
                    'Expected dict value for argument "fill_values" '
                )
            user_column_list = list(self.fill_values.keys())
            for column in user_column_list:
                if column not in col_list:
                    raise ArgumentsError(
                        f"Column {column} does not exist in dataframe"
                    )
        if self.cat_cols is not None:
            if type(self.cat_cols) is not list:
                raise TypeError('Expected list for argument "cat_cols"')
            for col in self.cat_cols:
                if col not in col_list:
                    raise ArgumentsError(
                        f"Column {col} does not exist in dataframe"
                    )


    # function to drop all rows with nan values
    def __drop_all_rows_with_null_values(self):
        self.new_train = self.train_df.dropna()
        if self.test_df is not None:
            self.new_test = self.test_df.dropna()
            return self.new_train, self.new_test
        return self.new_train, None

    # function to drop a particular column
    def __drop_column_with_null_values(self):

        self.new_train = self.train_df.drop(self.drop_cols, axis=1)
        if self.test_df is not None:
            self.new_test = self.train_df.drop(self.drop_cols, axis=1)
            return self.new_train, self.new_test
        return self.new_train, None

    # function to fill the missing values with mean or median as per the arguments passed
    def __fill_missing_with_mean_or_median(self, col_list, method):
        self.new_train = self.train_df
        if len(col_list) == 0:
            col_list = self.train_df.columns
            if method == "median":
                for new_train_col in col_list:
                    if self.new_train.dtypes[new_train_col] in dtypeList:
                        self.new_train[new_train_col].fillna(
                            self.new_train[new_train_col].median(), inplace=True
                        )
            else:
                for new_train_col in col_list:
                    if self.new_train.dtypes[new_train_col] in dtypeList:
                        self.new_train[new_train_col].fillna(
                            self.new_train[new_train_col].mean(), inplace=True
                        )

            if self.test_df is not None:
                if self.fill_missing == "median":
                    for new_test_col in self.new_test:
                        if self.new_test.dtypes[new_test_col] in dtypeList:
                            self.new_test[new_test_col].fillna(
                                self.new_test[new_test_col].median(), inplace=True
                            )
                else:
                    for new_test_col in self.new_test:
                        if self.new_test.dtypes[new_test_col] in dtypeList:
                            self.new_test[new_test_col].fillna(
                                self.new_test[new_test_col].mean(), inplace=True
                            )
                return self.new_train, self.new_test
            return self.new_train, None
        else:
            if method == "median":
                for new_train_col in col_list:
                        self.new_train[new_train_col].fillna(
                            self.new_train[new_train_col].median(), inplace=True
                        )
            else:
                for new_train_col in col_list:
                        self.new_train[new_train_col].fillna(
                            self.new_train[new_train_col].mean(), inplace=True
                        )

            if self.test_df is not None:
                if self.fill_missing == "median":
                    for new_test_col in col_list:
                            self.new_test[new_test_col].fillna(
                                self.new_test[new_test_col].median(), inplace=True
                            )
                else:
                    for new_test_col in col_list:
                            self.new_test[new_test_col].fillna(
                                self.new_test[new_test_col].mean(), inplace=True
                            )
                return self.new_train, self.new_test

            return self.new_train, None

    # function to fill columns containing null values with the character supplied by the user
    def __fill_values_columns(self):
        self.new_train = self.train_df
        for column in list(self.fill_values.keys()):
            self.new_train[column].fillna(
                self.fill_values[column], inplace=True
            )
        if self.test_df is not None:
            for column in list(self.fill_values.keys()):
                self.new_test[column].fillna(
                    self.fill_values[column], inplace=True
                )
                return self.new_train, self.new_test
        return self.new_train, None

    # function to remove or fill null values in categorical columns
    def __categoricalnull(self):
        self.new_train = self.train_df
        for col in self.cat_cols:
            if self.replace_cat_nulls is not None:
                self.new_train[col].fillna(self.replace_cat_nulls, inplace=True)
            else:
                self.new_train.dropna(axis=0, subset=[col], inplace=True)
        if self.test_df is not None:
            self.new_test = self.test_df
            for col in self.cat_cols:
                if self.replace_cat_nulls is not None:
                    self.new_test[col].fillna(
                        self.replace_cat_nulls, inplace=True
                    )
                else:
                    self.new_test.dropna(axis=0, subset=[col], inplace=True)
            self.test_df = self.new_test
        self.train_df = self.new_train

    # main function
    def execute(self, params):

        """Function that handles null values in the supplied dataframe and returns a new dataframe. If no user parameters are supplied, the rows containing null values are dropped by default.

        :param train_df: Input dataframe
                  Should not be ``None``
        :type train_df: pandas.core.frames.DataFrame

        :param test_df: Input dataframe
                  Should not be ``None``
        :type test_df: pandas.core.frames.DataFrame

        :param cat_cols: List containing the names of categorical columns
        :type cat_cols: list

        :param replace_cat_nulls: The value which will replace null values in the categorical columns
        :type replace_cat_nulls: str

        :param drop_cols: List of column names of columns to be dropped
        :type drop_cols: list

        :param fill_missing: List of format [[col_name],'method'] to indicate the method (``mean``/``median``) to be applied on specified col_list
        :type fill_missing: "mean" | "median"

        :param fill_values: Column and value mapping, where the key is the column name and value is the custom value to be filled in place of null values
        :type fill_values: dict

        """

        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "replace_cat_nulls" in params.keys():
            self.replace_cat_nulls = params["replace_cat_nulls"]
        if "drop_cols" in params.keys():
            self.drop_cols = params["drop_cols"]
        if "fill_missing" in params.keys():
            self.fill_missing = params["fill_missing"]
        if "fill_values" in params.keys():
            self.fill_values = params["fill_values"]


        self.__validate_input()

        self.__categoricalnull()
        if (
            self.drop_cols is not None
        ):
            (
                self.train_df,
                self.test_df,
            ) = self.__drop_column_with_null_values()

        if (
            self.fill_missing is not None
        ):
            (
                self.train_df,
                self.test_df,
            ) = self.__fill_missing_with_mean_or_median(self.fill_missing[0],self.fill_missing[1])

        if (
            self.fill_values is not None
        ):
            self.train_df, self.test_df = self.__fill_values_columns()

        (
            self.final_train,
            self.final_test,
        ) = self.__drop_all_rows_with_null_values()
        params["train_df"] = self.final_train
        params["test_df"] = self.final_test
