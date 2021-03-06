class HandleOutlier:

    """Class for handling outliers on its own or according to users needs.

     Private methods
    _ _ _ _ _ _ _ _ _ _

    __return_quartiles() : returns the 5% and 95% mark in the distribution
                          of data(Values above are default values)

     Public Methods
    _ _ _ _ _ _ _ _ _

    handle_outliers() : Takes in the dataset as input, finds the quartiles
                       and returns the dataset within the interquartile
                       range. Function run only on int64 and float64
                       specified columns.

    """

    def __init__(self):

        """Function to initialize a few parameters used to make the process
        run without human interaction.

        Parameters to be entered by the user include :
            -The dataset
            -cols: Any specific columns in dataset the user wants to
                   remove outliers from. If not entered columns will be
                   selected based on their dtype(int and float only) and
                   outliers will be removed from them
            -removeoutliers : default = True
            -replace : default = False => if user wants to replace
                       outliers with -999 everywhere instead of
                       removing them
            -q1 : specify the starting range (Default is 0.05)
            -q3 : specify the end of the range (Default is 0.95)

        """
        self.train_df = None
        self.cols = []
        self.remove_outliers = True
        self.replace = False
        self.quartiles = {}
        self.first_quartile = 0.05
        self.third_quartile = 0.95

    def __return_quartiles(self, col):
        # return the quartile range or q1 and q3 values for the column passed as parameter
        train_df = self.train_df
        q1 = train_df[col].quantile(self.first_quartile)
        q1 = round(q1)
        q3 = train_df[col].quantile(self.third_quartile)
        q3 = round(q3)
        self.quartiles[col] = [q1, q3]

    def handle_outliers(self, params):

        self.train_df = params["train_df"]
        if "cols" in params.keys():
            self.cols = params["cols"]
        if "removeoutliers" in params.keys():
            self.remove_outliers = params["removeoutliers"]
        if "replace" in params.keys():
            self.replace = params["replace"]
        if "q1" in params.keys():
            self.first_quartile = params["q1"]
        if "q3" in params.keys():
            self.third_quartile = params["q3"]

        # parameters till now: train_df, cols, removeoutliers, replace
        train_df = self.train_df
        if (
            self.remove_outliers
        ):  # if user has marked removeoutliers = True and wants outliers removed..
            if len(self.cols) >= 1:
                for col in self.cols:
                    self.__return_quartiles(col)
                for col in self.cols:
                    q = self.quartiles[col]
                    q1 = q[0]
                    q3 = q[1]
                    train_df = train_df[(train_df[col] >= q1)]
                    train_df = train_df[(train_df[col] <= q3)]

        # if removeoutliers = False and replace=True ie user wants outliers replaced by a value to indicate these
        # are outliers
        elif self.replace:
            if len(self.cols) >= 1:
                for col in self.cols:
                    self.__return_quartiles(col)
                for col in self.cols:
                    q = self.quartiles[col]
                    q1 = q[0]
                    q3 = q[1]
                    train_df[(train_df[col] < q1)] = -999
                    train_df[(train_df[col] > q3)] = -999
        print(self.quartiles)
        self.train_df = train_df
        params["train_df"] = self.train_df
