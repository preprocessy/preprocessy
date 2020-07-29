import pandas as pd
import numpy as np

class EncodeData:
    """  A class to encode categorical and ordinal features
    """


    def __init__(self, train_df, params):
        self.train_df = train_df
        self.cat_cols = []
        self.ord_cols = []
        self.ord_dict = {}
        if 'cat_cols' in params.keys():
            self.cat_cols = params['cat_cols']
        if 'ord_cols' in params.keys():
            self.ord_cols = params['ord_cols']
        if 'ord_dict' in params.keys():
            self.ord_dict = params['ord_dict']
        
    def __encode_categorical(self):
        if len(self.cat_cols)>=1:
            for col in self.cat_cols:
                self.train_df[col + str('Encoded')] = pd.factorize(self.train_df[col])[0]
                self.train_df[col + str('Encoded')
                              ] = self.train_df[col + str('Encoded')].astype('category')
        else:
            rows = self.train_df.shape[0]
            rows = 0.5*rows
            for col in self.train_df.columns:
                if '$' in self.train_df[col][0] or self.train_df[col].str.contains(',').any():
                    self.train_df[col] = self.train_df[col].apply(
                            lambda x: x.replace('$','').replace(',','')).astype('float')
                elif isinstance(self.train_df[col][0], int) and self.train_df[col].nunique() < rows:
                    self.cat_cols.append(col)
                elif isinstance(self.train_df[col][0], str) and self.train_df[col].nunique() > rows:
                    self.cat_cols.append(col)
                elif isinstance(self.train_df[col][0], float):
                    continue
            for col in self.cat_cols:
                self.train_df[col + str('Encoded')] = pd.factorize(self.train_df[col])[0]
                self.train_df[col + str('Encoded')
                              ] = self.train_df[col + str('Encoded')].astype('category')
        
        return self.train_df

    def __encode_ordinal(self):
        print(self.ord_cols,self.ord_dict)

    def encode(self):
        if self.ord_cols:
            self.__encode_ordinal()
        train = self.__encode_categorical()
        return train
        



'''
categorical : string('Teacher','Student'), int,
exclude : float, bool, datetime

pandas dtypes: 
    object - string or mixed => $1200 * 12,000
    int *
    float
    bool *
    datetime
    timestamp
    category = >final output
'''
