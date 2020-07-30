from preprocessy.encoding import EncodeData
import pandas as pd

ord_dict = {"Profession": {"Student": 1, "Teacher": 2, "HOD": 3}}

params = {"ord_dict": ord_dict}

train_csv = pd.read_csv("datasets/encoding/testnew.csv")
# print(train_csv.dtypes)
# train_csv = train_csv.drop(['Unnamed: 5','Unnamed: 6'],axis=1)
k = EncodeData(train_df=train_csv, params=params)
train = k.encode()

print(train)
