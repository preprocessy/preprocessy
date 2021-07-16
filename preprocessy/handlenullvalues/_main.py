import warnings

import pandas as pd

from ..exceptions import ArgumentsError


class NullValuesHandler:

    """Class for handling null values

    Parameters
    ---------------
    df : pandas.core.frames.DataFrame
         The input dataframe

    drop : boolean
           Controlling whether to drop rows/columns with null values or not.

    fill_missing : string
                   can be "mean" or "median", determines the value to be filled in place of null values.

    fill_values : dict
                  Column and value mapping, where the key is the column name and value is the custom value to be filled in place of null values

    column_list : list
                  List of column names which have to be dropped, if this parameter is not used, the dropping of values will occur on rows

    Private Methods
    ---------------

    __validate_input() : validates the input

    __drop_all_rows_with_null_values() : function to drop all rows with nan values

    __drop_column_with_null_values(column_name) : function to drop a particular column

    __fill_missing_with_mean_or_median() : function to fill the missing values with mean or median as per the arguments passed

    __fill_values_columns() : function to fill columns with null values with user specified value for corresponding columns

    Public Methods
    --------------

    execute(params) : Main function that performs the operations on supplied dataframe and returns a new dataframe

    """

    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.drop = None
        self.cat_cols = []
        self.replace_cat_nulls = None
        self.fill_missing = None
        self.fill_values = None
        self.column_list = None
        self.new_df = None
        self.final_train = None
        self.final_test = None
        self.final_df = None

    def __repr__(self):
        return f"NullValuesHandler(train_df=None, test_df=None, drop={self.drop}, fill_missing={self.fill_missing}, fill_values={self.fill_values})"

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
        if (
            self.drop is None
            and self.fill_missing is None
            and self.fill_values is None
        ):
            raise ArgumentsError(
                "Expected one argument apart from dataframe, received none"
            )

        if self.drop is not None:
            if self.fill_missing is not None or self.fill_values is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one"
                    " argument"
                )
            elif type(self.drop) is not bool:
                raise TypeError('Expected boolean value for argument "drop" ')
            elif self.drop is False:
                warnings.warn(
                    "Drop is False, thus no operation will be performed on"
                    " dataframe, please specify drop=True ",
                    UserWarning,
                )

        if self.fill_missing is not None:
            if self.drop is not None or self.fill_values is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one"
                    " argument"
                )
            elif type(self.fill_missing) is not str:
                raise TypeError(
                    'Expected string value for argument "fill_missing" '
                )
            self.fill_missing = self.fill_missing.lower()
            if self.fill_missing not in ["mean", "median"]:
                raise ArgumentsError('Allowed argument is "mean" or "median" ')

        if self.fill_values is not None:
            if self.drop is not None or self.fill_missing is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one"
                    " argument"
                )
            elif type(self.fill_values) is not dict:
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
        if self.column_list is not None:
            if not isinstance(self.column_list, list):
                raise TypeError('Expected List for argument "column_list"')
            if self.drop and len(self.column_list) == 0:
                warnings.warn(
                    '"column_list" is empty, no columns will be dropped. If'
                    ' you want to drop rows, do not pass "column_list" in the'
                    " arguments or pass None",
                    UserWarning,
                )
            if len(self.column_list) != 0:
                for c in self.column_list:
                    if c not in col_list:
                        raise ArgumentsError(
                            f'Column "{c}" does not exist in dataframe'
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
        self.new_train = self.train_df.drop(self.column_list, axis=1)
        if self.test_df is not None:
            self.new_test = self.train_df.drop(self.column_list, axis=1)
            return self.new_train, self.new_test
        return self.new_train, None

    # function to fill the missing values with mean or median as per the arguments passed
    def __fill_missing_with_mean_or_median(self):
        self.new_train = self.train_df
        if self.fill_missing == "median":
            self.new_train.fillna(self.new_train.median(), inplace=True)
        else:
            self.new_train.fillna(self.new_train.mean(), inplace=True)

        if self.test_df is not None:
            if self.fill_missing == "median":
                self.new_test.fillna(self.new_test.median(), inplace=True)
            else:
                self.new_test.fillna(self.new_test.mean(), inplace=True)
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

        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "replace_cat_nulls" in params.keys():
            self.replace_cat_nulls = params["replace_cat_nulls"]
        if "drop" in params.keys():
            self.drop = params["drop"]
        if "fill_missing" in params.keys():
            self.fill_missing = params["fill_missing"]
        if "fill_values" in params.keys():
            self.fill_values = params["fill_values"]
        if "column_list" in params.keys():
            self.column_list = params["column_list"]

        # automatic = drop rows with null values
        if (
            self.drop is None
            and self.column_list is None
            and self.fill_missing is None
            and self.fill_values is None
        ):
            self.drop = True

        self.__validate_input()

        self.__categoricalnull()
        if (
            self.drop is True
            and self.fill_missing is None
            and self.fill_values is None
            and self.column_list is None
        ):
            (
                self.final_train,
                self.final_test,
            ) = self.__drop_all_rows_with_null_values()

        elif (
            self.fill_missing is not None
            and self.drop is None
            and self.fill_values is None
        ):
            (
                self.final_train,
                self.final_test,
            ) = self.__fill_missing_with_mean_or_median()

        elif (
            self.fill_values is not None
            and self.drop is None
            and self.fill_missing is None
        ):
            self.final_train, self.final_test = self.__fill_values_columns()

        elif (
            self.drop is True
            and self.column_list is not None
            and len(self.column_list) != 0
        ):

            (
                self.final_train,
                self.final_test,
            ) = self.__drop_column_with_null_values()

        else:
            self.final_train = self.train_df
            self.final_test = self.test_df

        self.train_df = self.final_train
        self.test_df = self.final_test
        (
            self.final_train,
            self.final_test,
        ) = self.__drop_all_rows_with_null_values()
        params["train_df"] = self.final_train
        params["test_df"] = self.final_test
