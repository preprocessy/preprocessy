from handlecategorical import EncodeData
from pandas import DataFrame
import pandas as pd

ord_dict = {
    'Student' : 1,
    'Teacher' : 2,
    'HOD' : 3
}

params = {
    'ord_cols': ['Profession'],
    'ord_dict': ord_dict
}

train_csv = pd.read_csv('test.csv')
train_csv = train_csv.drop(['Unnamed: 4','Unnamed: 5','Unnamed: 6'],axis=1)

k = EncodeData(train_csv,params=params)
train = k.encode()

print(train)
