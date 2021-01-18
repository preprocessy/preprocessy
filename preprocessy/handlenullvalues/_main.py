import warnings
import pandas as pd
from .errors import ArgumentsError


class NullValuesHandler:

    """Class for handling null values

    Parameters
    ---------------
    df : pandas.core.frames.DataFrame
         The input dataframe

    drop : boolean
           Controlling whether to drop rows/columns with null values or not

    fill_missing : string
                   can be "mean" or "median", determines the value to be filled in place of null values

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

    execute() : Main function that performs the operations on supplied dataframe and returns a new dataframe

    """

    def __init__(self, df=None, drop=None, fill_missing=None, fill_values=None, column_list=None):
        self.df = df
        self.drop = drop
        self.fill_missing = fill_missing
        self.fill_values = fill_values
        self.column_list = column_list
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

        col_list = list(self.df)
        if self.fill_values is not None:
            if self.drop is not None or self.fill_missing is not None:
                raise ArgumentsError(
                    "Received more than one arguments, expected atmost one argument"
                )
            elif type(self.fill_values) is not dict:
                raise TypeError(f'Expected dict value for argument "fill_values" ')
<<<<<<< HEAD
=======
            col_list = list(self.df)
>>>>>>> 6181a2b (add option to drop columns)
            user_column_list = list(self.fill_values.keys())
            for column in user_column_list:
                if column not in col_list:
                    raise ArgumentsError(f"Column {column} does not exist in dataframe")

        if self.column_list is not None:
            if not isinstance(self.column_list,list):
                raise TypeError(f"Expected List for argument \"column_list\"")
<<<<<<< HEAD
            if self.drop and len(self.column_list) == 0:
                warnings.warn("\"column_list\" is empty, no columns will be dropped. If you want to drop rows, do not pass \"column_list\" in the arguments or pass None",UserWarning)
            if len(self.column_list) != 0:
                for c in self.column_list:
                    if c not in col_list:
=======
            if self.drop is not None and self.drop and len(self.column_list) == 0:
                warnings.warn("\"column_list\" is empty, no columns will be dropped. If you want to drop rows, do not pass \"column_list\" in the arguments or pass None",UserWarning)
            if len(self.column_list) != 0:
                cols=list(self.df)
                for c in self.column_list:
                    if c not in cols:
>>>>>>> 6181a2b (add option to drop columns)
                        raise ArgumentsError(f"Column \"{c}\" does not exist in dataframe")


    # function to drop all rows with nan values
    def __drop_all_rows_with_null_values(self):
        self.new_df = self.df.dropna()
        return self.new_df

    # function to drop a particular column
    def __drop_column_with_null_values(self):
        self.new_df = self.df.drop(self.column_list, axis=1)
        return self.new_df

    # function to fill the missing values with mean or median as per the arguments passed
    def __fill_missing_with_mean_or_median(self):
        self.new_df = self.df
        if self.fill_missing == "median":
            self.new_df.fillna(self.new_df.median(),inplace=True)
        else:
            self.new_df.fillna(self.new_df.mean(),inplace=True)
        return self.new_df

    # function to fill columns containing null values with the character supplied by the user
    def __fill_values_columns(self):
        self.new_df = self.df
        for column in list(self.fill_values.keys()):
            self.new_df[column].fillna(self.fill_values[column], inplace=True)
        return self.new_df

    def execute(self):

        if (
            self.drop is not None
            and self.fill_missing is None
            and self.fill_values is None
            and self.drop != False
            and self.column_list is None
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
        
        elif (
<<<<<<< HEAD
            self.drop == True
=======
            self.drop is not None
            and self.drop != False
>>>>>>> 6181a2b (add option to drop columns)
            and self.column_list is not None
            and len(self.column_list) != 0
        ):
            self.final_df = self.__drop_column_with_null_values()

        else:
            return self.df

        return self.final_df
