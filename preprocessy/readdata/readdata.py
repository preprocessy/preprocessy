import csv
import pandas as pd

#Code to read files and select columns.

with open('Filepath', 'rb') as csv_file:  #Reading the file
    csv_reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames

    class field:
        def __init__(self, **fields):
            self.__dict__.update(**fields)

        def __repr__(self):  # Added to make printing instances show their contents.
            fields = ', '.join(('{}={!r}'.format(fieldname, getattr(self, fieldname))
                                   for fieldname in fieldnames))
            return('{}({})'.format(self.__class__.__name__, fields))

    List = [field(**row) for row in reader]

print(List)

class info:
  #For information on the data and its type.
  csv_file.info()

class drop:
  #Dropping columns
  csv_file.drop(csv_file.columns[csv_file.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
