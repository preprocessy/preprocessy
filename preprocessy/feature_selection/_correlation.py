import pandas as pd


class Correlation:
    """Class for finding correlation between features

    Private Methods
    ---------------

    __validate__input() : validates input received by find()

    Public Methods
    --------------

    find() : Finds correlation between feature columns

    """

    def __validate_input(self, X, threshold):
        """Function to validate inputs received by find()

        Parameters
        ----------

        X : pandas.core.frames.DataFrame
            Input dataframe.

        threshold : float
                    Two columns are said to be correlated if their value lies beyond the threshold

        Returns
        -------

        Raises errors for invalid inputs

        """

        if X is None:
            raise ValueError("Feature dataframe should not be of None")
        if not isinstance(X, pd.core.frame.DataFrame):
            raise TypeError(
                "Feature dataframe is not a valid dataframe.\nExpected object"
                " type: pandas.core.frame.DataFrame"
            )
        if threshold < 0 or threshold > 1:
            raise ValueError(
                "Threshold value should lie between 0 and 1. Value passed is"
                f" {threshold}"
            )

    def find(self, X, threshold=0.8):
        """Function to find highly correlated and columns with no correlation in the given dataset

        Parameters
        ----------

        X : pandas.core.frames.DataFrame
            Input dataframe.

        threshold : float
                    Two columns are said to be correlated if their value lies beyond the threshold.

        Returns
        -------

        results : list<tuple<string,string,float,string>>
                    A list of tuples where each tuple contains names of the columns, their correlation value and sign of correlation.

        """
        self.__validate_input(X, threshold)

        corr = X.corr()
        columns = corr.columns
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[1]):
                curr = corr.iloc[i, j]
                if j != i and (
                    curr >= threshold or curr <= -threshold or curr == 0
                ):
                    sign = "Positive Correlation"
                    if curr == 0:
                        sign = "No correlation"
                    elif curr < 0:
                        sign = "Negative Correlation"
                    yield (columns[i], columns[j], curr, sign)
