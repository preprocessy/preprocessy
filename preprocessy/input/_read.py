import errno
import os

import pandas as pd


class Reader(object):
    """Standard Reader Class that serves to read and load numeric data into pandas dataframe.

    The file extensions allowed are: .csv, .xls, .xlxs, .xlsm, .xlsb, .odf, .ods, .odt
    """

    def __init__(self):
        self.excel_extensions = [
            "xls",
            "xlsx",
            "xlsm",
            "xlsb",
            "odf",
            "ods",
            "odt",
        ]
        self.train_df_path = None
        self.test_df_path = None

    def _validate_input(self, file_name):
        if type(file_name) is not str:
            raise TypeError(
                f"Argument file_name should be of str type. Received {type(file_name)}"
            )
        else:
            self.file_name = file_name

    def __read_file_util(self, file_name):

        self._validate_input(file_name)
        df = None

        if ".csv" in self.file_name:
            df = pd.read_csv(self.file_name)
            if df is None:
                df = pd.read_csv(self.file_name, delimiter=";")
        elif ".tsv" in self.file_name:
            df = pd.read_csv(self.file_name, sep="\t")
        elif self.file_name.split(".")[-1] in self.excel_extensions:
            df = pd.read_excel(self.file_name)
        else:
            raise ValueError(
                f'Unsupported filetype. Supported extensions include [.csv, .tsv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods and .odt]. Received file of type .{self.file_name.split(".")[-1]}'
            )

        if df is not None:
            df.drop(
                df.columns[df.columns.str.contains("unnamed", case=False)],
                axis=1,
                inplace=True,
            )
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.file_name
            )

        return df

    def read_file(self, params):
        """Function to take the train and test dataframe paths and load it in pandas dataframe

        :param train_df_path: Path that points to the train dataset(Extension can be any of the above listed).
                        Should not be ``None``.
        :type train_df_path: str

        :param test_df_path: Path that points to the test dataset(Extension can be any of the above listed).
        :type test_df_path: str
        """
        if "train_df_path" in params.keys():
            self.train_df_path = params["train_df_path"]
        if "test_df_path" in params.keys():
            self.test_df_path = params["test_df_path"]

        params["train_df"] = self.__read_file_util(self.train_df_path)
        if self.test_df_path:
            params["test_df"] = self.__read_file_util(self.test_df_path)
