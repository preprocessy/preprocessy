from handleoutlier import HandleOutlier
from pandas import DataFrame

params = {
    'cols':['Test'],
    'removeoutliers' :True,
    'replace' :False,
    'q1' : 0.10,
    'q3' : 0.90
}
train_df = [1,2,3,100,0,3,6,4,20]
train_df = DataFrame(data=train_df, columns=['Test'])
k = HandleOutlier(train_df,params=params)
train = k.handle_outliers()
print(train_df)
print(train)