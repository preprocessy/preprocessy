import pandas as pd

from ..exceptions import ArgumentsError


class Scaler:
    def __init__(self):

        self.train_df = None
        self.test_df = None
        self.type = "StandardScaler"
        self.columns = []
        self.is_combined = False
        self.threshold = None
        self.new_train_df = None
        self.new_test_df = None
        self.final_train_df = None
        self.final_test_df = None
        self.cat_cols = None
        self.target_label = None

    def __repr__(self):
        return f"Scaler(type={self.type}, is_combined={self.is_combined}, threshold={self.threshold})"

    def isNumeric(self, column):
        # i => int (signed), u => unsigned int, f => float, c => complex
        return column.dtype.kind in "iufc"

    def __validate_input(self):
        if self.train_df is None:
            raise ValueError(
                "Feature train dataframe should not be of None type"
            )

        if type(self.train_df) is not pd.core.frame.DataFrame:
            raise TypeError(
                "Feature train dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )

        if self.test_df is not None:
            if type(self.test_df) is not pd.core.frame.DataFrame:
                raise TypeError(
                    "Feature test dataframe is not a valid dataframe.\nExpected object"
                    " type: pandas.core.frame.DataFrame"
                )

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
            if not isinstance(self.columns, list):
                raise TypeError(
                    f"Expected list type for argument columns, got {type(self.columns)}"
                )
            column_list = list(self.train_df.keys())
            for column in self.columns:
                if type(column) != str:
                    raise TypeError(
                        f"Expected str type column, got {type(column)}"
                    )
                if column not in column_list:
                    raise ArgumentsError(
                        f"Column {column} does not exist in dataframe"
                    )
        else:
            cols = list(self.train_df.keys())
            for col in cols:
                if self.isNumeric(self.train_df[col]):
                    self.columns.append(col)

        if self.threshold is not None:
            if type(self.threshold) is not dict:
                raise TypeError(
                    f"Expected dict type threshold, got {type(column)}"
                )
            for column in self.threshold.keys():
                if column not in list(self.train_df.keys()):
                    raise ArgumentsError(
                        f"Column {column} does not exist in dataframe"
                    )

        if self.cat_cols is not None:
            if not isinstance(self.cat_cols, list):
                raise TypeError(
                    f"Expected list type for argument categorical_columns, got {type(self.cat_cols)}"
                )

        if not isinstance(self.target_label, str):
            raise TypeError(
                f"Expected str type for argument target_col, got {type(self.columns)}"
            )

        self.new_train_df = self.train_df
        self.new_test_df = self.test_df

    def __min_max_scaler_helper(self, df):
        new_df = df.copy()
        to_be_dropped_columns = list()
        if self.cat_cols is not None:
            to_be_dropped_columns = self.cat_cols
        to_be_dropped_columns.append(self.target_label)
        if not self.is_combined:
            for column in self.columns:
                if column in to_be_dropped_columns:
                    continue
                if not self.isNumeric(df[column]):
                    raise TypeError(
                        f"Unexpected datatype of column, {type(column)}"
                    )
                cur_col = df[column]
                max = cur_col.max()
                min = cur_col.min()
                cur_col = (cur_col - min) / (max - min)
                new_df[column] = cur_col
        else:
            temp_df = new_df.drop(columns=to_be_dropped_columns)
            max = temp_df.to_numpy().max()
            min = temp_df.to_numpy().min()
            for column in self.columns:
                if column in to_be_dropped_columns:
                    continue
                if not self.isNumeric(df[column]):
                    raise TypeError(
                        f"Unexpected datatype of column, {type(column)}"
                    )
                new_df[column] = (temp_df[column] - min) / (max - min)
        return new_df

    def __min_max_scaler(self):
        if self.train_df is not None:
            self.new_train_df = self.__min_max_scaler_helper(self.train_df)
        if self.test_df is not None:
            self.new_test_df = self.__min_max_scaler_helper(self.test_df)
        return self.new_train_df, self.new_test_df

    def __binary_scaler_helper(self, df):
        new_df = df.copy()
        to_be_dropped_columns = list()
        if self.cat_cols is not None:
            to_be_dropped_columns = self.cat_cols
        to_be_dropped_columns.append(self.target_label)
        for column in self.columns:
            if not self.isNumeric(df[column]):
                raise TypeError(
                    f"Unexpected datatype of column, {type(column)}"
                )
            if column in to_be_dropped_columns:
                continue
            cur_thresh = 0
            if self.threshold is not None:
                if column in self.threshold.keys():
                    cur_thresh = self.threshold[column]
            new_df[column] = df[column].apply(
                lambda val: 0 if val <= cur_thresh else 1
            )
        return new_df

    def __binary_scaler(self):
        if self.train_df is not None:
            self.new_train_df = self.__binary_scaler_helper(self.train_df)
        if self.test_df is not None:
            self.new_test_df = self.__binary_scaler_helper(self.test_df)
        return self.new_train_df, self.new_test_df

    def __standard_scaler_helper(self, df):
        new_df = df.copy()
        to_be_dropped_columns = list()
        if self.cat_cols is not None:
            to_be_dropped_columns = self.cat_cols
        to_be_dropped_columns.append(self.target_label)
        if not self.is_combined:
            for column in self.columns:
                if column in to_be_dropped_columns:
                    continue
                if not self.isNumeric(df[column]):
                    raise TypeError(
                        f"Unexpected datatype of column, {type(column)}"
                    )
                cur_col = df[column]
                mean = cur_col.mean()
                std = cur_col.std()
                cur_col = (cur_col - mean) / std
                new_df[column] = cur_col
        else:
            temp_df = new_df.drop(columns=to_be_dropped_columns)
            mean = temp_df.stack().mean()
            std = temp_df.stack().std()
            for column in self.columns:
                if column in to_be_dropped_columns:
                    continue
                if not self.isNumeric(df[column]):
                    raise TypeError(
                        f"Unexpected datatype of column, {type(column)}"
                    )
                new_df[column] = (temp_df[column] - mean) / std
        return new_df

    def __standard_scaler(self):
        if self.train_df is not None:
            self.new_train_df = self.__standard_scaler_helper(self.train_df)
        if self.test_df is not None:
            self.new_test_df = self.__standard_scaler_helper(self.test_df)
        return self.new_train_df, self.new_test_df

    def execute(self, params):
        """Method for scaling the columns in a dataset

        :param train_df: Input dataframe
                  Should not be ``None``
        :type train_df: pandas.core.frames.DataFrame

        :param test_df: Input dataframe
                  Should not be ``None``
        :type test_df: pandas.core.frames.DataFrame

        :param type: The type of Scaler to be used
        :type type: "MinMaxScaler" | "BinaryScaler" | "StandardScaler"

        :param columns: List of columns in the dataframe
        :type columns: list

        :param cat_cols: List containing the names of categorical columns
        :type cat_cols: list

        :param target_label: Name of the Target Column. This parameter is needed to ensure that the target column
                            isn't scaled
        :type target_label: str

        :param is_combined: Parameter to determine whether columns should be scaled together as a group
        :type is_combined: bool

        :param threshold: Dictionary of threshold values where the key is the column name and the value is the threshold for that column.
        :type threshold: dict


        """
        if "type" in params.keys():
            self.type = params["type"]
        if "columns" in params.keys():
            self.columns = params["columns"]
        if "is_combined" in params.keys():
            self.is_combined = params["is_combined"]
        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "threshold" in params.keys():
            self.threshold = params["threshold"]
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "target_label" in params.keys():
            self.target_label = params["target_label"]

        self.__validate_input()

        if self.type == "MinMaxScaler":
            self.final_train_df, self.final_test_df = self.__min_max_scaler()
        elif self.type == "BinaryScaler":
            self.final_train_df, self.final_test_df = self.__binary_scaler()
        elif self.type == "StandardScaler":
            self.final_train_df, self.final_test_df = self.__standard_scaler()

        params["train_df"] = self.final_train_df
        params["test_df"] = self.final_test_df
