import numbers

import numpy as np

from ..utils import num_of_samples


class KFold:
    """Class for splitting input data into K-folds. Split
    dataset into K consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the ``k - 1`` remaining
    folds form the training set.

    :param n_splits: Number of folds. Must be at least 2, defaults to 5
    :type n_splits: int

    :param shuffle: Whether to shuffle the data before splitting
                    into folds, defaults to ``False``
    :type shuffle: bool

    :param random_state: Random state used for shuffling, defaults to ``None``
    :type random_state: int

    """

    def __init__(self, n_splits=5, shuffle=False, random_state=None) -> None:

        if not isinstance(n_splits, numbers.Integral):
            raise ValueError(
                f"Number of folds must be an integer. {n_splits} of type"
                f" {type(n_splits)} was passed"
            )

        if n_splits <= 1:
            raise ValueError(
                "K-fold cross-validation requires at least one train/test"
                " split by setting n_splits=2 or more, received"
                f" n_splits={n_splits}."
            )

        if not isinstance(shuffle, bool):
            raise ValueError(
                f"shuffle must be boolean value. Received {shuffle}"
            )

        if not isinstance(random_state, numbers.Integral):
            raise ValueError(
                f"Random state must be an integer. {random_state} of type"
                f" {type(random_state)} was passed"
            )

        if not shuffle and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since shuffle is False."
                " You should leave random_state to its default (None), or set"
                " shuffle=True.",
            )

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def __repr__(self):
        return f"KFold(n_splits={self.n_splits}, shuffle={self.shuffle}, random_state={self.random_state})"

    def split(self, X, y=None):
        """Generate indices to split data into training and test set.

        :param X: Input dataframe, may or may not consist of the target label.
                  Should not be ``None``
        :type X: pandas.core.frames.DataFrame

        :param y: Target label series. If ``None`` then ``X`` consists target label,
                  defaults to ``None``
        :type y: pandas.core.series.Series, optional

        :yield: The training set indices for that split
        :rtype: ndarray

        :yield: The testing set indices for that split
        :rtype: ndarray

        :raises ValueError: If ``n_splits > n_samples``, then a ``ValueError`` is raised

        """

        n_samples = num_of_samples(X)

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
