import numpy as np
import pandas as pd

from ..exceptions import ArgumentsError


class Scaler:
    def __init__(self):
        """Class for Scaling the columns

        Private Methods
        ---------------

        __validate_input() : validates the input

        __min_max_scaler() : function to scale the supplied columns on the basis of Min-Max-Scaling technique

        __binary_scaler() : function to scale the supplied columns on the basis of binary Scaling

        __standard_scaler() : function to standardise the supplied columns

        Note: For more information on these scaling techniques read .scaling_tecniques.txt

        Public Methods
        --------------

        execute() : Main function that performs the operations on supplied dataframe and returns a new dataframe

        """

        self.df = None
        self.type = None
        self.columns = None
        self.is_combined = False
        self.critical_value = 0
        self.new_df = None
        self.final_df = None

    def __validate_input(self):
        if self.df is None:
            raise ValueError("Feature dataframe should not be of None type")

        if type(self.df) is not pd.core.frame.DataFrame:
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )

        if self.type is None:
            raise ValueError("Feature type should not be of None type")
        else:
            if type(self.type) is not str:
                raise TypeError('Expected string value for argument "type" ')
            if self.type not in [
                "MinMaxScaler",
                "BinaryScaler",
                "StandardScaler",
            ]:
                raise ArgumentsError(
                    'Allowed argument for type is "MinMaxScaler" or'
                    f' "BinaryScaler" or "StandardScaler", got {self.type}'
                )

        if self.columns is not None:
            if type(self.columns) is not list:
                raise TypeError(
                    f"Expected list type for argument columns, got {type(self.columns)}"
                )
            column_list = list(self.df.keys())
            for column in self.columns:
                if type(column) != str:
                    raise TypeError(
                        f"Expected str type column, got {type(column)}"
                    )
                if column not in column_list:
                    raise ArgumentsError(
                        f"Column {column} does not exist in dataframe"
                    )

        self.new_df = self.df

    def __min_max_scaler(self):
        if not self.is_combined:
            for column in self.columns:
                new_col = np.array([val for val in self.df[column]])
                max = np.max(new_col)
                min = np.min(new_col)
                diff = max - min
                for i in range(len(new_col)):
                    new_col[i] = (new_col[i] - min) / diff
                self.new_df[column] = new_col
        else:
            new_combined_arr = []
            for column in self.columns:
                for ele in self.df[column]:
                    new_combined_arr.append(ele)
            new_combined_arr = np.array(new_combined_arr)
            max = np.max(new_combined_arr)
            min = np.min(new_combined_arr)
            diff = max - min
            for column in self.columns:
                new_col = [val for val in self.df[column]]
                for i in range(len(new_col)):
                    new_col[i] = (new_col[i] - min) / diff
                self.new_df[column] = new_col

        return self.new_df

    def __binary_scaler(self):
        for column in self.columns:
            new_arr = [val for val in self.df[column]]
            for i in range(len(new_arr)):
                if new_arr[i] <= self.critical_value:
                    new_arr[i] = 0
                else:
                    new_arr[i] = 1
            self.new_df[column] = new_arr

        return self.new_df

    def __standard_scaler(self):
        if not self.is_combined:
            for column in self.columns:
                mean = self.new_df[column].mean()
                std = self.new_df[column].std()
                new_col = [ele for ele in self.df[column]]
                for i in range(len(new_col)):
                    new_col[i] = (new_col[i] - mean) / std
                self.new_df[column] = new_col
        else:
            new_combined_arr = []
            for column in self.columns:
                for ele in self.df[column]:
                    new_combined_arr.append(ele)
            new_combined_arr = np.array(new_combined_arr)
            mean = new_combined_arr.mean()
            std = new_combined_arr.std()
            for column in self.columns:
                new_col = [val for val in self.df[column]]
                for i in range(len(new_col)):
                    new_col[i] = (new_col[i] - mean) / std
                self.new_df[column] = new_col

        return self.new_df

    def execute(self, params):

        if "df" in params.keys():
            self.df = params["df"]
        if "type" in params.keys():
            self.type = params["type"]
        if "columns" in params.keys():
            self.columns = params["columns"]
        if "is_combined" in params.keys():
            self.is_combined = params["is_combined"]
        if "critical_value" in params.keys():
            self.critical_value = params["critical_value"]

        self.__validate_input()

        if self.type == "MinMaxScaler":
            self.final_df = self.__min_max_scaler()
        elif self.type == "BinaryScaler":
            self.final_df = self.__binary_scaler()
        elif self.type == "StandardScaler":
            self.final_df = self.__standard_scaler()

        params["df"] = self.final_df
