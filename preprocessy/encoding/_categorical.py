import pandas as pd
import numpy as np
import warnings


class EncodeData:
    """  A class to encode categorical and ordinal features
    """

    def __init__(self, params, target_label=None, train_df=None, test_df=None):
        """
        params for initializing:
        params = {
            cat_cols: list of categorical columns as perceived by the user
            ord_dict: dictionary of dictionary of key and value for ordinal columns in the said format
                        dict = {
                            'col1' = dict_col1,
                            'col2' = dict_col2
                        }
                        where dict_col1 = {
                            'key1': 'value1',
                            ...
                            ...
                            ...
                        }
            train_df: training dataset (pandas.core.frames.DataFrame)
            test_df: test dataset (pandas.core.frames.DataFrame)
            target_value: name of target column so as to ensure that target columns isn't present in
                          train_df
        }
        """
        self.test_df = test_df
        self.train_df = train_df
        self.target_label = target_label
        self.cat_cols = None
        self.ord_dict = None
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "ord_dict" in params.keys():
            self.ord_dict = params["ord_dict"]

    def __validate_inputs(self):
        """
        function to validate inputs 
        - check if training dataset is given(crucial)
        -check if test df is provided then whether the number of columns are same . 
        -check if target variable is still not removed from training set
        -check if mapping dictionary is provided for every ordinal column specified 

        """
        if self.train_df is None:
            raise ValueError("Dataframe cannot be null")

        if self.test_df is not None and self.train_df.shape[1] != self.test_df.shape[1]:
            raise KeyError("Target variable in still present in one of the datasets or the number of columns in both test and train are not equal. Rectify")

        # target_label should not be in list of columns
        if self.target_label is None or (
            self.target_label in self.train_df.columns
            and (self.cat_cols is None or self.target_label in self.cat_cols)
        ):
            warnings.warn(
                "target_label may get encoded. Please remove target_label from dataframe or provide explicit list of columns for encoding.",
                UserWarning,
            )
        for col,mapping in self.ord_dict.items():
            for key in mapping:
                if mapping[key] is None or mapping[key] == {}:
                    raise ValueError(
                        f"Expected a weight mapping for ordinal columns {key}. Received {self.ord_dict[key]}"
                    )

    def __encode_categorical_util(self):
        """
        helper function to encode categorical columns using pd.factorize. Encoding will not affect the 
        original columns. Instead a mew column will be made with the name 'col_nameEncoding'.
        """
        for col in self.cat_cols:
            if col in self.train_df and col not in self.ord_cols:
                if self.test_df is not None:
                    self.test_df[col + str("Encoded")] = pd.factorize(self.train_df[col])[
                        0
                    ]
                    self.test_df[col + str("Encoded")] = self.test_df[
                        col + str("Encoded")
                    ].astype("category")
                self.train_df[col + str("Encoded")] = pd.factorize(self.train_df[col])[
                    0
                ]
                self.train_df[col + str("Encoded")] = self.train_df[
                    col + str("Encoded")
                ].astype("category")


    def __encode_categorical(self):
        """
        function to find out which columns may be categorical . All such columns will have their title
        added to self.cat_cols which the user can use to identify which columns has the code identified as 
        categorical. During encoding the map formed for encoding won't be sorted. Sort will be added in v2
        if demanded.
        """
        if self.cat_cols is not None:
            self.__encode_categorical_util()
        else:
            rows = self.train_df.shape[0]
            rows = 0.5 * rows
            self.cat_cols = []
            for col in self.train_df.columns:
                if (
                    "$" in self.train_df[col][0]
                    or self.train_df[col].str.contains(",").any()
                ):
                    self.train_df[col] = (
                        self.train_df[col]
                        .apply(lambda x: x.replace("$", "").replace(",", ""))
                        .astype("float")
                    )
                elif (
                    pd.to_datetime(self.train_df[col], errors="coerce").isnull().sum()
                    <0.7*len(train_df[col])
                ):
                    self.train_df[col] = pd.to_datetime(
                        self.train_df[col], errors="coerce"
                    )
                elif (
                    isinstance(self.train_df[col][0], int)
                    and self.train_df[col].nunique() < rows
                ):
                    self.cat_cols.append(col)
                elif (
                    isinstance(self.train_df[col][0], str)
                    and self.train_df[col].nunique() < rows
                ):
                    self.cat_cols.append(col)
            self.__encode_categorical_util()

        return True

    def __encode_ordinal(self):
        """
        function to encode ordinal columns as provided by user in form of a dictionary. Format for the 
        parameter is specified at the top during initialization.
        """
        self.ord_cols=[]
        for col, mapping in self.ord_dict.items():
            for key, value in mapping.items():
                if key in self.train_df.columns:
                    if self.test_df is not None:
                        self.test_df[key + str("Encoded")
                                      ] = self.test_df[key].map(value)
                        self.test_df[key + str("Encoded")
                                    ] = self.test[key + str("Encoded")].astype("category")
                        self.ord_cols.append(key)

                    self.train_df[key + str("Encoded")] = self.train_df[key].map(value)
                    self.train_df[key + str("Encoded")
                                ] = self.train_df[key + str("Encoded")].astype("category")
                    self.ord_cols.append(key)

    def encode(self):
        """
        function to encode columns
        -ordinal: mapping dictionary needed and mapping will be done accordingly
        -categorical: cat_cols can be provided by user else the code will determine which
                      columns can be categorical.
        After encoding all new columns made will have dtype='category'
        As an additional feature the code determines to find columns where integers exist 
        with $ or commas example- Money columns. The commas and $ are removed and the column has its dtype
        converted to int64. However this feature won't be activated if categorical columns are mentioned by
        the user. 

        Returns
        _ _ _ _ _

        self.train_df: Modified training set
        self.test_df: Modified test set
        cat_cols: Columns which were taken as categorical
        """
        self.__validate_inputs()
        if self.ord_dict:
            self.__encode_ordinal()
        self.__encode_categorical()
        return (self.train_df,self.test_df,self.cat_cols)
