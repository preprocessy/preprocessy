import errno
import os
from datetime import datetime

import pandas as pd


class Reader(object):
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
        start = datetime.now()
        if "train_df_path" in params.keys():
            self.train_df_path = params["train_df_path"]
        if "test_df_path" in params.keys():
            self.test_df_path = params["test_df_path"]

        params["train_df"] = self.__read_file_util(self.train_df_path)
        if self.test_df_path:
            params["test_df"] = self.__read_file_util(self.test_df_path)
        end = datetime.now()
        duration = end - start
        print(
            "-------------Completed reading and loading the .csv file in "
            + str(duration)
            + " -------------"
        )
