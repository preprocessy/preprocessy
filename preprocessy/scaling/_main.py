import pandas as pd
import math_funcs.math_func as math_func
from errors import ArgumentsError


class Scaler:
    def __init__(
        self, df=None, type=None, columns=None, is_combined=False, critical_value=0
    ):
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

        self.df = df
        self.type = type
        self.columns = columns
        self.is_combined = is_combined
        self.critical_value = critical_value
        self.new_df = None
        self.__validate_input()
        self.final_df = None

    def __validate_input(self):
        if self.df is None:
            raise ValueError("Feature dataframe should not be of None type")

        if type(self.df) is not pd.core.frame.DataFrame:
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object type: pandas.core.frame.DataFrame"
            )

        if self.type is None:
            raise ValueError("Feature type should not be of None type")
        else:
            if type(self.type) is not str:
                raise TypeError(f'Expected string value for argument "type" ')
            if self.type not in ["MinMaxScaler", "BinaryScaler", "StandardScaler"]:
                raise ArgumentsError(
                    f'Allowed argument for type is "MinMaxScaler" or "BinaryScaler" or "StandardScaler", got {self.type}'
                )

        if self.columns is not None:
            column_list = list(self.df.keys())
            for column in self.columns:
                if type(column) != str:
                    raise TypeError(f"Expected str type column, got {type(column)}")
                if column not in column_list:
                    raise ArgumentsError(f"Column {column} does not exist in dataframe")

        self.new_df = self.df

    def __min_max_scaler(self):
        if not self.is_combined:
            for column in self.columns:
                new_col = [val for val in self.df[column]]
                max = math_func.max(new_col)
                min = math_func.min(new_col)
                diff = max - min
                for i in range(len(new_col)):
                    new_col[i] = (new_col[i] - min) / diff
                self.new_df[column] = new_col
        else:
            new_mega_arr = []
            for column in self.columns:
                for ele in self.df[column]:
                    new_mega_arr.append(ele)
            max = math_func.max(new_mega_arr)
            min = math_func.min(new_mega_arr)
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
            new_mega_arr = []
            for column in self.columns:
                for ele in self.df[column]:
                    new_mega_arr.append(ele)
            mean = math_func.mean(new_mega_arr)
            std = math_func.stdev(new_mega_arr)
            for column in self.columns:
                new_col = [val for val in self.df[column]]
                for i in range(len(new_col)):
                    new_col[i] = (new_col[i] - mean) / std
                self.new_df[column] = new_col

        return self.new_df

    def execute(self):
        if self.type == "MinMaxScaler":
            self.final_df = self.__min_max_scaler()
        elif self.type == "BinaryScaler":
            self.final_df = self.__binary_scaler()
        elif self.type == "StandardScaler":
            self.final_df = self.__standard_scaler()

        return self.final_df


"""df = pd.read_csv(
    'C:/Users/yash/Downloads/37281_63530_bundle_archive/test_data9.csv')
df = df.dropna()
df = Scaler(df=df, type="StandardScaler", columns=[
            'Id', 'ParentId'], is_combined=True, critical_value=20)
df = df.execute()
print(df)"""
