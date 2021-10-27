import warnings

from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype


class Parser:
    def __init__(self):
        self.train_df = None
        self.test_df = None
        self.target_label = None
        self.cat_cols = None
        self.ord_cols = []
        self.ord_dict = None

    def __validate_input(self):
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

    def __get_cat_cols(self):
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
                        .apply(lambda x: x.replace("$", "").replace(",", ""))
                        .astype("float")
                    )
                elif (
                    is_numeric_dtype(self.train_df[col])
                    or is_string_dtype(self.train_df[col])
                ) and self.train_df[col].dropna().nunique() < rows:
                    self.cat_cols.append(col)
            else:
                continue
        return self.cat_cols

    def parse_dataset(self, params):
        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "ord_dict" in params.keys():
            self.ord_dict = params["ord_dict"]
        if "target_label" in params.keys():
            self.target_label = params["target_label"]

        self.__validate_input()

        if "cat_cols" in params.keys():
            self.cat_cols = params["cat_cols"]
            for col in self.cat_cols:
                if col not in self.train_df or (
                    self.test_df is not None and col not in self.test_df
                ):
                    raise ValueError(
                        f"Column {col} is not present in the given dataset"
                    )
            return

        if self.ord_dict:
            self.ord_cols = [k for k in self.ord_dict.keys()]
            params["ord_cols"] = self.ord_cols

        params["cat_cols"] = self.__get_cat_cols()
