import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype


class Encoder:
    """A class to encode categorical and ordinal features"""

    def __init__(self):
        self.test_df = None
        self.train_df = None
        self.target_label = None

        self.cat_cols = None
        self.ord_dict = None
        self.ord_cols = []
        self.one_hot = False

    def __repr__(self):
        return f"Encoder(target_label={self.target_label} ,train_df=None, test_df=None, cat_cols=None, ord_dict=None, one_hot={self.one_hot})"

    def __validate_inputs(self):
        """
        function to validate inputs
        - check if training dataset is given(crucial)
        - check if test df is provided then whether the number of columns are same
        - check if target variable is still not removed from training set
        - check if mapping dictionary is provided for every ordinal column specified

        """
        if self.train_df is None:
            raise ValueError("Dataframe cannot be null")

        if (
            self.test_df is not None
            and self.train_df.shape[1] != self.test_df.shape[1]
        ):
            raise KeyError(
                "Target variable in still present in one of the datasets or"
                " the number of columns in both test and train are not equal."
            )

        # target_label should not be in list of columns
        if self.target_label is None or (
            self.target_label in self.train_df.columns
            and (self.cat_cols is None or self.target_label in self.cat_cols)
        ):
            warnings.warn(
                "target_label may get encoded. Please remove target_label from"
                " dataframe or provide explicit list of columns for encoding.",
                UserWarning,
            )

        if self.ord_dict is not None:
            for key, mapping in self.ord_dict.items():
                if mapping is None or mapping == {}:
                    raise ValueError(
                        f"Expected a weight mapping for ordinal column {key}."
                        f" Received {self.ord_dict[key]}"
                    )

    def __encode_categorical_util(self):
        """
        Helper function to encode categorical columns using pd.factorize. Encoding will not affect the
        original columns. Instead a new column will be made with the name 'col_nameEncoding'.
        """
        cat = []
        # cat = self.cat_cols
        for col in self.cat_cols:
            if (
                col in self.train_df
                and col + str("Encoded") not in self.ord_cols
            ):
                if self.test_df is not None:
                    self.test_df[col + str("Encoded")] = pd.factorize(
                        self.test_df[col]
                    )[0]
                    self.test_df[col + str("Encoded")] = self.test_df[
                        col + str("Encoded")
                    ].astype("category")
                self.train_df[col + str("Encoded")] = pd.factorize(
                    self.train_df[col]
                )[0]
                self.train_df[col + str("Encoded")] = self.train_df[
                    col + str("Encoded")
                ].astype("category")
                cat.append(str(col + str("Encoded")))
        self.cat_cols += cat

    def __encode_one_hot_util(self):
        """
        Helper function to one-hot encode categorical columns using pd.get_dummies. Encoding will not affect the
        original columns. Instead a new column will be made with the name 'col_labelname'.
        """
        for col in self.cat_cols:
            if (
                col in self.train_df
                and col + str("Encoded") not in self.ord_cols
            ):
                if self.test_df is not None:
                    self.test_df = pd.concat(
                        [
                            self.test_df,
                            pd.get_dummies(
                                self.test_df[col], prefix=col
                            ).astype("category"),
                        ],
                        axis=1,
                    )
                self.train_df = pd.concat(
                    [
                        self.train_df,
                        pd.get_dummies(self.train_df[col], prefix=col).astype(
                            "category"
                        ),
                    ],
                    axis=1,
                )

    def __encode_categorical(self):
        """
        Function to find out which columns may be categorical. All such columns will have their title
        added to self.cat_cols which the user can use to identify which columns the code has identified as
        categorical. During encoding the map formed for encoding won't be sorted. Sort will be added in v2
        if demanded.
        """
        if self.cat_cols is None:
            rows = self.train_df.shape[0]
            rows = 0.09 * rows
            self.cat_cols = []
            for col in self.train_df.columns:
                if col not in self.ord_cols:
                    if (
                        self.train_df[col].dtype == "object"
                        and type(self.train_df[col][0]) == "str"
                    ) and (
                        "$" in self.train_df[col][0]
                        or self.train_df[col].str.contains(",").any()
                    ):
                        self.train_df[col] = (
                            self.train_df[col]
                            .apply(
                                lambda x: x.replace("$", "").replace(",", "")
                            )
                            .astype("float")
                        )
                    # elif pd.to_datetime(
                    #     self.train_df[col], errors="coerce"
                    # ).isnull().sum() < 0.7 * len(self.train_df[col]):
                    #     self.train_df[col] = pd.to_datetime(
                    #         self.train_df[col], errors="coerce"
                    #     )
                    elif (
                        is_numeric_dtype(self.train_df[col])
                        or is_string_dtype(self.train_df[col])
                    ) and self.train_df[col].dropna().nunique() < rows:
                        self.cat_cols.append(col)
                else:
                    continue

        if self.one_hot:
            self.__encode_one_hot_util()
        else:
            self.__encode_categorical_util()
        return

    def __encode_ordinal(self):
        """
        Function to encode ordinal columns as provided by user in the form of a dictionary. Format for the
        parameter is specified at the top during initialization.
        """
        for key, value in self.ord_dict.items():
            if key in self.train_df.columns:
                if self.test_df is not None:
                    self.test_df[key + str("Encoded")] = self.test_df[key].map(
                        value
                    )
                    self.test_df[key + str("Encoded")] = self.test_df[
                        key + str("Encoded")
                    ].astype("category")

                self.train_df[key + str("Encoded")] = self.train_df[key].map(
                    value
                )
                self.train_df[key + str("Encoded")] = self.train_df[
                    key + str("Encoded")
                ].astype("category")
                self.ord_cols.append(key + str("Encoded"))

    def encode(self, params):
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
            one_hot:  boolean parameter to detect if one_hot_encoding is to be done

            train_df    : training dataset (pandas.core.frames.DataFrame)
            test_df     : test dataset (pandas.core.frames.DataFrame)
            target_label: name of target column so as to ensure that target columns isn't present in
                          train_df
        }

        Function to encode columns
        - ordinal: mapping dictionary needed and mapping will be done accordingly
        - categorical: cat_cols can be provided by user else the code will determine which
                      columns can be categorical.
        - one_hot encoding: bool value can be provided by user for one_hot categorical encoding else
                      default is set to false.
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
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "target_label" in params.keys():
            self.target_label = params["target_label"]
        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
        if "ord_dict" in params.keys():
            self.ord_dict = params["ord_dict"]
        if "one_hot" in params.keys():
            if params["one_hot"] is True:
                self.one_hot = True

        self.__validate_inputs()
        if self.ord_dict:
            self.__encode_ordinal()
        self.__encode_categorical()

        params["train_df"] = self.train_df
        params["test_df"] = self.test_df
        params["cat_cols"] = self.cat_cols
        params["ord_cols"] = self.ord_cols
