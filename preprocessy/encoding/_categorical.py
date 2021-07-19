import warnings

import pandas as pd
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype


class Encoder:
    """Class to encode categorical and ordinal features.
    Categorical encoding options include: ``normal`` and ``one-hot``"""

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
        if self.target_label is None:
            warnings.warn(
                "Parameter 'target_label' is empty. If not provided and is present in dataframe, it may get encoded. "
                "To mitigate, provide the target_label from dataframe or provide explicit list of columns for encoding "
                "via the 'cat_cols' parameter",
                UserWarning,
            )
        if (
            self.target_label is not None
            and self.cat_cols is not None
            and (self.target_label in self.cat_cols)
        ):
            raise ValueError(
                f"Target column: {self.target_label} will be encoded. Remove it from cat_cols if in there."
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
            rows = 0.2 * rows
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
        Function to encode categorical or ordinal columns.

        :param train_df: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type train_df: pandas.core.frames.DataFrame

        :param test_df: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type test_df: pandas.core.frames.DataFrame

        :param target_label: Name of the Target Column. This parameter is needed to ensure that the target column
                            isn't identified as categorical and encoded.
        :type target_label: str

        :param cat_cols: List containing the column names to be encoded categorically
        :type cat_cols: list

        :param ord_dict: Dictionary with the the key as name of column to be encoded ordinally and the corresponding value is the dictionary containing the mapping.
        :type ord_dict: dict

        :param one_hot: This parameter takes True or False to indicate whether the user wants to encode using one-hot.
        :type one-hot: bool

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
