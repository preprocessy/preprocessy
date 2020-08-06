import pandas as pd
import os


class ReadData(object):
    def __init__(self, name_file):
        self.name_file = name_file
        self.df = self.__read_file()
        self.summary, self.stats = self.__read_summary()

    def __read_file(self):
        """Read the file content"""

        if ".csv" not in self.name_file:
            raise ValueError(
                f'Invalid filename. Expected path to file of type ".csv". Received {self.name_file}'
            )
        try:
            df = None
            df = pd.read_csv(self.name_file, index_col=0)
            df.drop(
                df.columns[df.columns.str.contains("unnamed", case=False)],
                axis=1,
                inplace=True,
            )
        except FileNotFoundError:
            print(f"{self.name_file} not found")
        finally:
            return df

    def __read_summary(self):
        """Read file summary"""

        summary = None
        stats = None
        try:
            summary = self.df.info()
            stats = self.df.describe()
        except FileNotFoundError:
            print(f"{self.name_file} not found")
        finally:
            return summary, stats

    def display_file(self):
        print(self.df)
        print(self.df.columns)

    def display_summary(self):
        print(self.summary)
        print(self.stats)
