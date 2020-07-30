import numpy as np
import pandas as pd


class Split:
    """ Class for resampling and splitting input data
    
    Private Methods
    ---------------

    __validate__input() : validates input received by train_test_split()

    Public Methods
    --------------

    train_test_split() : Splits input data into train and test sets

    """

    def __init__(self):
        pass

    def __validate_input(self, X, y, test_size, train_size, random_state):

        """ Function to validate inputs received by train_test_split

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

        random_state : int
            Seeding to be provided for shuffling before splitting.

        Returns
        -------

        train_size: float or int
            Returns default value of 0.7 if not provided any value.

        test_size: float or int
            Returns default value of 0.3 if not provided any value.
        
        """

        if X is None:
            raise ValueError(f"Feature dataframe should not be of None")

        if type(X) is not pd.core.frame.DataFrame:
            raise TypeError(
                f"Feature dataframe is not a valid dataframe.\nExpected object type: pandas.core.frame.DataFrame"
            )

        n_samples = X.shape[0]

        if y is not None:
            if n_samples != y.shape[0]:
                raise ValueError(
                    f"Number of samples of target label and feature dataframe unequal.\nSamples in feature dataframe: {X.shape[0]}\nSamples in target label: {y.shape[0]}"
                )
            if type(y) is not pandas.core.series.Series:
                raise TypeError(
                    f"Target label is not a valid dataframe.\nExpected object type: pandas.core.series.Series"
                )

        if test_size and train_size:
            if type(test_size) is not int or type(test_size) is not float:
                raise TypeError(f"test_size must be of type int or float")
            if type(train_size) is not int or type(train_size) is not float:
                raise TypeError(f"train_size must be of type int or float")
            if type(test_size) != type(train_size):
                raise TypeError(
                    f"Data types of test_size and train_size do not match.\ntest_size: {type(test_size)}.\ntrain_size: {type(train_size)}"
                )
            if type(test_size) is float and test_size + train_size != 1:
                raise ValueError(f"test_size + train_size should be equal to 1")
            elif type(test_size) is int and test_size + train_size != n_samples:
                raise ValueError(
                    f"test_size + train_size not equal to number of samples"
                )

        elif test_size:
            if type(test_size) is float and (test_size < 0 or test_size > 1):
                raise ValueError(f"test_size should be between 0 and 1")
            if type(test_size) is int and (test_size < 0 or test_size > n_samples):
                raise ValueError(f"test_size should be between 0 and {n_samples}")
            train_size = (
                1 - test_size if type(test_size) is float else n_samples - test_size
            )

        elif train_size:
            if type(train_size) is float and (train_size < 0 or train_size > 1):
                raise ValueError(f"train_size should be between 0 and 1")
            if type(train_size) is int and (train_size < 0 or train_size > n_samples):
                raise ValueError(f"train_size should be between 0 and {n_samples}")
            test_size = (
                1 - train_size if type(train_size) is float else n_samples - train_size
            )

        else:
            if y is None:
                test_size = 0.2
                train_size = 0.8
            else:
                features = len(X.columns)
                test_size = 1 / np.sqrt(features)
                train_size = 1 - test_size

        if type(random_state) is not int:
            raise TypeError(f"random_state should be of type int")

        return train_size, test_size

    def train_test_split(
        self, X=None, y=None, test_size=None, train_size=None, random_state=69
    ):
        """Performs train test split on the input data

        Parameters
        ----------
        X : pandas.core.frames.DataFrame
            Input dataframe, may or may not consist of the target label.
            Should not be None
        
        y : pandas.core.series.Series
            Target label series. If None then X consists target label

        test_size : float or int
            Size of test set after splitting. Can take values from 0 - 1 for float point values,
            0 - Number of samples for integer values. Is complementary to train size.

        train_size : float or int
            Size of train set after splitting. Can take values from 0 - 1 for float point values,
            0 - Number of samples for integer values. Is complementary to test size.

        random_state : int
            Seeding to be provided for shuffling before splitting.

        Returns
        -------

        If target label provided

            X_train : pandas.core.frames.DataFrame

            y_train : pandas.core.series.Series

            X_test : pandas.core.frames.DataFrame

            y_test : pandas.core.series.Series

        Else

            train : pandas.core.frames.DataFrame

            test : pandas.core.frames.DataFrame

        """

        train_size, test_size = self.__validate_input(
            X, y, test_size, train_size, random_state
        )

        np.random.seed(random_state)

        if y is not None:
            df = pd.concat(X, y, axis=1)
        else:
            df = X

        df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)

        if type(test_size) is float:
            index = int(test_size * len(df))
            train = df.iloc[index:]
            test = df.iloc[:index]
        else:
            train = df.iloc[test_size:]
            test = df.iloc[:test_size]

        if y is not None:
            y_train = train[y.name]
            X_train = train.drop([y.name], axis=1)
            y_test = test[y.name]
            X_test = test.drop([y.name], axis=1)
            return X_train, y_train, X_test, y_test

        return train, test
