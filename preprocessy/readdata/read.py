import pandas as pd
import os


class ReadData(object):
    def __init__(self, name_file):
        self.name_file = name_file
        self.df = self.read_file()
        self.sf = self.read_summary()

    def read_file(self):
        """Read the file content"""

        try:
            self.df = pd.read_csv(self.name_file, index_col=0)
            # self.df.drop(self.df.filter(regex="Unname"),axis=1, inplace=True)
            self.df.drop(self.df.columns[self.df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        except IndexError:
            print("Error: Wrong file name")
            sys.exit(2)
        return self.df

    def display_file(self):
        print(self.df)

    def read_summary(self):
        """Read file summary"""

        try:
            self.sf = pd.read_csv(self.name_file)
        except IndexError:
            print("Error: Wrong file name")
            sys.exit(2)
        return self.sf

    def display_summary(self):
        self.sf.info()

