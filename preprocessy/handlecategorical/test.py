from handlecategorical import EncodeData
from pandas import DataFrame

params = {
    'cat_cols': ['Test']
}
train_df = [0,0,1,3,4,1,2,2]
train_df = DataFrame(data=train_df, columns=['Test'])
k = EncodeData(train_df, params=params)
train = k.encode_categorical()
# print(train)
