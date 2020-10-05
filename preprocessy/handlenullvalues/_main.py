import warnings
import pandas as pd
from .errors import ArgumentsError


class NullValuesHandler:

    """Class for handling null values

    Private Methods
    ---------------

    __validate_input() : validates the input

    __drop_all_rows_with_null_values() : function to drop all rows with nan values

    __drop_column_with_null_values(column_name) : function to drop a particular column

    __fill_missing_with_mean_or_median() : function to fill the missing values with mean or median as per the arguments passed

    __fill_values_columns() : function to fill columns with null values with user specified value for corresponding columns

    Public Methods
    --------------

    execute() : Main function that performs the operations on supplied dataframe and returns a new dataframe

    """

    def __init__(self, df=None, drop=None, fill_missing=None, fill_values=None):
        self.df = df
        self.drop = drop
        self.fill_missing = fill_missing
        self.fill_values = fill_values
        self.__validate_input()
        self.new_df = None
        self.final_df = None

    def __validate_input(self):
        if self.df is None:
            raise ValueError("Feature dataframe should not be of None type")

        if type(self.df) is not pd.core.frame.DataFrame:
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object type: pandas.core.frame.DataFrame"
            )

        if self.drop is None and self.fill_missing is None and self.fill_values is None:
            raise ArgumentsError(
                "Expected one argument apart from dataframe, received none"
            )

        if self.drop is not None:
            if self.fill_missing is not None or self.fill_values is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one argument"
                )
            elif type(self.drop) is not bool:
                raise TypeError(f'Expected boolean value for argument "drop" ')
            elif self.drop == False:
                warnings.warn(
                    "Drop is False, thus no operation will be performed on dataframe, please specify drop=True ",
                    UserWarning,
                )

        if self.fill_missing is not None:
            if self.drop is not None or self.fill_values is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one argument"
                )
            elif type(self.fill_missing) is not str:
                raise TypeError(f'Expected string value for argument "fill_missing" ')
            self.fill_missing = self.fill_missing.lower()
            if self.fill_missing not in ["mean", "median"]:
                raise ArgumentsError('Allowed argument is "mean" or "median" ')

        if self.fill_values is not None:
            if self.drop is not None or self.fill_missing is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one argument"
                )
            elif type(self.fill_values) is not dict:
                raise TypeError(f'Expected dict value for argument "fill_values" ')
            column_list = list(self.df)
            user_column_list = list(self.fill_values.keys())
            for column in user_column_list:
                if column not in column_list:
                    raise ArgumentsError(f"Column {column} does not exist in dataframe")

    # function to drop all rows with nan values
    def __drop_all_rows_with_null_values(self):
        self.new_df = self.df.dropna()
        return self.new_df

    # function to drop a particular column
    def __drop_column_with_null_values(self, column_name):
        self.new_df = self.df.drop([column_name], axis=1)
        return self.new_df

    # function to fill the missing values with mean or median as per the arguments passed
    def __fill_missing_with_mean_or_median(self):
        self.new_df = self.df
        if self.fill_missing == "median":
            self.new_df = self.new_df.fillna(self.new_df.median())
        else:
            self.new_df = self.new_df.fillna(self.new_df.mean())
        return self.new_df

    # function to fill columns containing null values with the character supplied by the user
    def __fill_values_columns(self):
        self.new_df = self.df
        print(list(self.fill_values.keys()))
        for column in list(self.fill_values.keys()):
            print(column)
            self.new_df[column].fillna(self.fill_values[column], inplace=True)
        return self.new_df

    def execute(self):

        if (
            self.drop is not None
            and self.fill_missing is None
            and self.fill_values is None
            and self.drop != False
        ):
            self.final_df = self.__drop_all_rows_with_null_values()

        elif (
            self.fill_missing is not None
            and self.drop is None
            and self.fill_values is None
        ):
            self.final_df = self.__fill_missing_with_mean_or_median()

        elif (
            self.fill_values is not None
            and self.drop is None
            and self.fill_missing is None
        ):
            self.final_df = self.__fill_values_columns()

        else:
            return self.df

        return self.final_df
