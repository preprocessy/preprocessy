import pandas as pd
import os
import errno


class ReadData(object):
    def __init__(self, file_name):
        self.excel_extensions = ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]
        self._validate_input(file_name)
        self.__read_file()
        self.summary, self.stats = self.__read_summary()

    def _validate_input(self, file_name):
        if type(file_name) is not str:
            raise TypeError(
                f'Argument file_name should be of str type. Received type {str(type(file_name)).split(" ")[-1].split(">")[0]}'
            )
        else:
            self.file_name = file_name

    def __read_file(self):
        """Read the file content"""

        self.df = None

        if ".csv" in self.file_name:
            self.df = pd.read_csv(self.file_name, index_col=0)
            if self.df is None:
                self.df = pd.read_csv(self.file_name, index_col=0, delimiter=";")
        elif ".tsv" in self.file_name:
            self.df = pd.read_csv(self.file_name, sep="\t")
        elif self.file_name.split(".")[-1] in self.excel_extensions:
            self.df = pd.read_excel(self.file_name)
        else:
            raise ValueError(
                f'Unsupported filetype. Supported extensions include [.csv, .tsv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods and .odt]. Received file of type .{self.file_name.split(".")[-1]}'
            )

        if self.df is not None:
            self.df.drop(
                self.df.columns[self.df.columns.str.contains("unnamed", case=False)],
                axis=1,
                inplace=True,
            )
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.file_name
            )

    def __read_summary(self):
        """Read file summary"""

        summary = None
        stats = None
        if self.df is not None:
            summary = self.df.info()
            stats = self.df.describe()
        return summary, stats

    def display_file(self):
        print(self.df)
        print(self.df.columns)

    def display_summary(self):
        print(self.summary)
        print(self.stats)
