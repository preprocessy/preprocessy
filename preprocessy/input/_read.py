import pandas as pd
import os

class ReadData(object):
    def __init__(self, file_name):
        self.excel_extensions = ["xls", "xlsx", "xlsm", "xlsb", "odf", "ods", "odt"]
        self.file_name = file_name
        self.df = self.__read_file()
        self.summary, self.stats = self.__read_summary()

    def __read_file(self):
        """Read the file content"""

        if ".csv" not in self.file_name:
            raise ValueError(
                f'Invalid filename. Expected path to file of type ".csv". Received {self.file_name}'
            )
        try:
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

            self.df.drop(
                self.df.columns[self.df.columns.str.contains("unnamed", case=False)],
                axis=1,
                inplace=True,
            )

        except FileNotFoundError:
            print(f"{self.file_name} not found")

        finally:
            return self.df

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
