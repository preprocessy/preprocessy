import numpy as np
import pandas as pd

from ..utils import num_of_samples


class KFold:
    """Class for splitting input data into K-folds. Split
    dataset into K consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the ``k - 1`` remaining
    folds form the training set.
    """

    def __init__(self):
        self.df = None
        self.n_splits = 5
        self.shuffle = False
        self.random_state = None

    def __validate_input(self):
        if self.df is None:
            raise ValueError("Feature dataframe should not be None")

        if not isinstance(self.df, pd.core.frame.DataFrame):
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )

        if not isinstance(self.n_splits, int):
            raise ValueError(
                f"Number of folds must be an integer. Received {self.n_splits} of type"
                f" {type(self.n_splits)}."
            )

        if self.n_splits <= 1:
            raise ValueError(
                "K-fold cross-validation requires at least one train/test"
                " split by setting n_splits=2 or more, received"
                f" n_splits={self.n_splits}."
            )

        if not isinstance(self.shuffle, bool):
            raise ValueError(
                f"shuffle must be boolean value. Received {self.shuffle} of type {type(self.shuffle)}."
            )

        if self.random_state and not isinstance(self.random_state, int):
            raise ValueError(
                f"Random state must be an integer. Received {self.random_state} of type"
                f" {type(self.random_state)}."
            )

        if not self.shuffle and self.random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since shuffle is False."
                " You should leave random_state to its default (None), or set"
                " shuffle=True.",
            )

    def __repr__(self):
        return f"KFold(n_splits={self.n_splits}, shuffle={self.shuffle}, random_state={self.random_state})"

    def split(self, params):
        """Generate indices to split data into training and test set.

        :param train_df: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type train_df: pandas.core.frames.DataFrame

        :param test_df: Input dataframe, may or may not consist of the target label.
        :type test_df: pandas.core.frames.DataFrame

        :param n_splits: Number of folds. Must be at least 2
        :type n_splits: int, default = 5

        :param shuffle: Whether to shuffle the data before splitting
                        into folds
        :type shuffle: bool, default = False

        :param random_state: Random state used for shuffling
        :type random_state: int, default = None

        :yield: The training set indices for that split
        :rtype: ndarray

        :yield: The testing set indices for that split
        :rtype: ndarray

        :raises ValueError: If ``n_splits > n_samples``, then a ``ValueError`` is raised

        """

        if "train_df" in params.keys():
            self.df = params["train_df"]
        if "test_df" in params.keys():
            self.df = pd.concat([self.df, params["test_df"]])
        if "n_splits" in params.keys():
            self.n_splits = params["n_splits"]
        if "shuffle" in params.keys():
            self.shuffle = params["shuffle"]
        if "random_state" in params.keys():
            self.random_state = params["random_state"]

        self.__validate_input()

        n_samples = num_of_samples(self.df)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have number of splits {self.n_splits} > number of"
                f" samples {n_samples}"
            )

        indices = np.arange(n_samples)
        for test_indices in self.__iter_test_indices(n_samples):
            train_indices = indices[np.logical_not(test_indices)]
            test_indices = indices[test_indices]
            yield train_indices, test_indices

    def __iter_test_indices(self, n_samples):
        """Generate masked indices for the current fold that is going to serve as test set.

        Parameters
        ----------

        n_samples : int Number of samples in the dataset

        Yields
        ------

        mask : ndarray Array of size n_samples. If a sample belongs to fold that is going to be the test set then mask has value 1 else 0.

        """
        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)

        fold_sizes = np.full(
            self.n_splits, n_samples // self.n_splits, dtype=int
        )
        fold_sizes[: n_samples % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            mask = np.zeros(n_samples, dtype=bool)
            mask[indices[start:stop]] = True
            yield mask
            current = stop

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator

        :return: Returns the number of splitting iterations in the cross-validator.
        :rtype: int
        """
        return self.n_splits
