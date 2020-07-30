import pandas as pd
import numpy as np
import warnings


class EncodeData:
    """  A class to encode categorical and ordinal features
    """

    def __init__(self, params, target_label=None, train_df=None):
        self.train_df = train_df
        self.target_label = target_label
        self.cat_cols = None
        self.ord_dict = None
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "ord_dict" in params.keys():
            self.ord_dict = params["ord_dict"]

    def __validate_inputs(self):
        if self.train_df is None:
            raise ValueError("Dataframe cannot be null")

        # target_label should not be in list of columns
        if self.target_label is None or (
            self.target_label in self.train_df.columns
            and (self.cat_cols is None or self.target_label in self.cat_cols)
        ):
            warnings.warn(
                "target_label may get encoded. Please remove target_label from dataframe or provide explicit list of columns for encoding.",
                UserWarning,
            )

        for key in self.ord_dict.keys():
            if self.ord_dict[key] is None or self.ord_dict[key] == {}:
                raise ValueError(
                    f"Expected a weight mapping for ordinal columns {key}. Received {self.ord_dict[key]}"
                )

    def __encode_categorical_util(self):
        for col in self.cat_cols:
            if col in self.train_df:
                self.train_df[col + str("Encoded")] = pd.factorize(self.train_df[col])[
                    0
                ]
                self.train_df[col + str("Encoded")] = self.train_df[
                    col + str("Encoded")
                ].astype("category")

    def __encode_categorical(self):
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
                    pd.to_datetime(self.train_df[col], errors="coerce").isnull().any()
                    != True
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
                    and self.train_df[col].nunique() > rows
                ):
                    self.cat_cols.append(col)
            self.__encode_categorical_util()

        return self.train_df

    def __encode_ordinal(self):
        for k, v in self.ord_dict.items():
            if k in self.train_df.columns:
                self.train_df[k] = self.train_df[k].map(v)

    def encode(self):
        self.__validate_inputs()
        if self.ord_dict:
            self.__encode_ordinal()
        train = self.__encode_categorical()
        return train


"""
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
"""
