import numpy as np
import pandas as pd

from ..exceptions import ArgumentsError
from ..utils import num_of_samples


class Split:
    """Class for splitting the dataset into train and test sets"""

    def __init__(self):
        self.df = None
        self.train_df = None
        self.target_label = None
        self.train_y = None
        self.test_df = None
        self.test_y = None
        self.test_size = None
        self.train_size = None
        self.random_state = None
        self.shuffle = False

    def __repr__(self):
        return f"Split(test_size={self.test_size}, train_size={self.train_size}, random_state={self.random_state})"

    def __validate_input(self):
        """Function to validate inputs received by train_test_split

        Parameters
        ----------

        X : pandas.core.frames.DataFrame
            Input dataframe, may or may not consist of the target label.

        y : pandas.core.series.Series
            Target label series. If None then X consists target label

        test_size : float or int
            Size of test set after splitting. Can take values from 0 - 1 for float point values,
            0 - Number of samples for integer values. Is complementary to train size.

        train_size : float or int
            Size of train set after splitting. Can take values from 0 - 1 for float point values,
            0 - Number of samples for integer values. Is complementary to test size.

        shuffle : bool, default = False
            Decides whether the data should be shuffled before splitting
        random_state : int
            Seeding to be provided for shuffling before splitting.

        Returns
        -------

        train_size: float or int
            Returns default value of 0.7 if not provided any value.

        test_size: float or int
            Returns default value of 0.3 if not provided any value.

        """

        if self.train_df is None:
            raise ValueError("Feature dataframe should not be None")

        if not isinstance(self.train_df, pd.core.frame.DataFrame):
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )

        n_samples = num_of_samples(self.train_df)

        if self.train_y is not None:
            if n_samples != self.train_y.shape[0]:
                raise ValueError(
                    "Number of samples of target label and feature dataframe"
                    " unequal.\nSamples in feature dataframe:"
                    f" {self.X.shape[0]}\nSamples in target label: {self.y.shape[0]}"
                )
            if not isinstance(self.train_y, pd.core.series.Series):
                raise TypeError(
                    "Target label is not a valid dataframe.\nExpected object"
                    " type: pandas.core.series.Series"
                )
        if self.test_size and self.train_size:
            if not isinstance(self.test_size, int) or not isinstance(
                self.test_size, float
            ):
                raise TypeError("test_size must be of type int or float")
            if not isinstance(self.train_size, int) or not isinstance(
                self.train_size, float
            ):
                raise TypeError("train_size must be of type int or float")
            if not isinstance(self.test_size, self.train_size):
                raise TypeError(
                    "Data types of test_size and train_size do not"
                    f" match.\ntest_size: {type(self.test_size)}.\ntrain_size:"
                    f" {type(self.train_size)}"
                )
            if (
                isinstance(self.test_size, float)
                and self.test_size + self.train_size != 1
            ):
                raise ValueError("test_size + train_size should be equal to 1")
            elif (
                isinstance(self.test_size, int)
                and self.test_size + self.train_size != n_samples
            ):
                raise ValueError(
                    "test_size + train_size not equal to number of samples"
                )

        elif self.test_size:
            if isinstance(self.test_size, float) and (
                self.test_size < 0 or self.test_size > 1
            ):
                raise ValueError("test_size should be between 0 and 1")
            if isinstance(self.test_size, int) and (
                self.test_size < 0 or self.test_size > n_samples
            ):
                raise ValueError(
                    f"test_size should be between 0 and {n_samples}"
                )
            self.train_size = (
                1 - self.test_size
                if isinstance(self.test_size, float)
                else n_samples - self.test_size
            )

        elif self.train_size:
            if isinstance(self.train_size, float) and (
                self.train_size < 0 or self.train_size > 1
            ):
                raise ValueError("train_size should be between 0 and 1")
            if isinstance(self.train_size, int) and (
                self.train_size < 0 or self.train_size > n_samples
            ):
                raise ValueError(
                    f"train_size should be between 0 and {n_samples}"
                )
            self.test_size = (
                1 - self.train_size
                if isinstance(self.train_size, float)
                else n_samples - self.train_size
            )

        else:
            if self.train_y is None:
                self.test_size = 0.2
                self.train_size = 0.8
            else:
                features = len(self.train_df.columns)
                self.test_size = float(1 / np.sqrt(features))
                self.train_size = 1 - self.test_size

        if not isinstance(self.shuffle, bool):
            raise TypeError(
                f"shuffle should be of type bool. Received {self.shuffle} of type {type(self.shuffle)}."
            )
        if self.random_state and not isinstance(self.random_state, int):
            raise TypeError(
                f"random_state should be of type int. Received {self.random_state} of type {type(self.random_state)}."
            )
        if self.random_state and not self.shuffle:
            raise ArgumentsError(
                f"random_state should be None when shuffle is set to False. Received {self.random_state} as random_state."
            )

    def train_test_split(self, params):
        """Performs train test split on the input data

        :param train_df: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type train_df: pandas.core.frames.DataFrame

        :param test_df: Input dataframe, may or may not consist of the target label.
        :type test_df: pandas.core.frames.DataFrame

        :param target_label: Name of the Target Column.
        :type target_label: str

        :param test_size: Size of test set after splitting. Can take values from
                          0 - 1 for float point values, 0 - Number of samples for
                          integer values. Is complementary to train size.
        :type test_size: float, int

        :param train_size: Size of train set after splitting. Can take values from
                           0 - 1 for float point values, 0 - Number of samples for
                           integer values. Is complementary to test size.
        :type train_size: float, int

        :param shuffle: Decides whether to shuffle data before splitting.
        :type shuffle: bool, default = False

        :param random_state: Seeding to be provided for shuffling before splitting.
        :type random_state: int

        The functions inserts the following into ``params`` -

        If target label is provided

        - **X_train** : pandas.core.frames.DataFrame

        - **y_train** : pandas.core.series.Series

        - **X_test** : pandas.core.frames.DataFrame

        - **y_test** : pandas.core.series.Series

        Else

        - **train**: pandas.core.frames.DataFrame

        - **test**: pandas.core.frames.DataFrame

        :raises ValueError: If the target column does not have a ``name`` property
                            ``ValueError`` is raised.

        """

        if "train_df" in params.keys():
            self.train_df = params["train_df"]
        if "test_df" in params.keys():
            self.test_df = params["test_df"]
        if "target_label" in params.keys():
            self.target_label = params["target_label"]
        if "test_size" in params.keys():
            self.test_size = params["test_size"]
        if "train_size" in params.keys():
            self.train_size = params["train_size"]
        if "shuffle" in params.keys():
            self.shuffle = params["shuffle"]
        if "random_state" in params.keys():
            self.random_state = params["random_state"]
        if self.target_label and self.test_df is not None:
            self.train_y = self.train_df[self.target_label]
            self.test_y = self.test_df[self.target_label]

        self.__validate_input()
        if self.test_df is not None and self.test_y is not None:
            params["X_train"] = self.train_df.drop([self.target_label], axis=1)
            params["X_test"] = self.test_df.drop([self.target_label], axis=1)
            params["y_train"] = self.train_y
            params["y_test"] = self.test_y

        elif self.test_df is not None:
            params["X_train"] = self.train_df
            params["X_test"] = self.test_df

        else:

            if self.shuffle and self.random_state:
                np.random.seed(self.random_state)

            if self.train_y is not None:
                self.df = pd.concat([self.train_df, self.train_y], axis=1)
            else:
                self.df = self.train_df

            if self.shuffle:
                self.df = self.df.iloc[
                    np.random.permutation(len(self.df))
                ].reset_index(drop=True)

            if isinstance(self.test_size, float):
                index = int(self.test_size * len(self.df))
                train = self.df.iloc[index:]
                test = self.df.iloc[:index]
            else:
                train = self.df.iloc[self.test_size :]
                test = self.df.iloc[: self.test_size]

            if self.train_y is not None:
                if not self.train_y.name:
                    raise ValueError(
                        f"Target column needs to have a name. ${self.train_y.name} was provided."
                    )
                y_train = train[self.train_y.name]
                X_train = train.drop([self.train_y.name], axis=1)
                y_test = test[self.train_y.name]
                X_test = test.drop([self.train_y.name], axis=1)
                params["X_train"] = X_train
                params["X_test"] = X_test
                params["y_train"] = y_train
                params["y_test"] = y_test

            else:
                params["X_train"] = train
                params["X_test"] = test
