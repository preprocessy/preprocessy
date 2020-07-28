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
        
    def encode_categorical(self):
        t = self.train_df.select_dtypes('number')
        print(t)
        if len(self.cat_cols)>=1:
            for col in self.cat_cols:
                self.train_df[col + str('Encoded')] = pd.factorize(self.train_df[col])[0]
                self.train_df[col + str('Encoded')
                              ] = self.train_df[col + str('Encoded')].astype('category')
        else:
            train_cat = self.train_df.select_dtypes('number')
            for col in train_cat.columns:
                self.train_df[col + str('Encoded')] = pd.factorize(self.train_df[col])[0]
                self.train_df[col + str('Encoded')
                              ] = self.train_df[col + str('Encoded')].astype('category')
        
        return self.train_df
        
