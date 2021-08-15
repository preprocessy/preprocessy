
Parameters for pipeline configuration
=====================================

The parameters for the pipeline are to be given in a ``dict`` or in a json file named config.json .
The paramters are listed below along with their meaning, expected dtype, defualt(if any).
Any parameter marked ``*`` are essential to the smooth working of the pipeline and are required.

- \*train_df_path(``str``): Path to the train dataset. Path should point to a file of approved extension.

- \*test_df_path(``str``): Path to the test dataset. Path should point to a file of approved extension. For list of allowed extensions see :py:mod:`preprocessy.input`

- target_label(``str``): Name of the target column.

- cat_cols(``list``): List of column names provided by user indicating these columns are to be encoded categorically. If ``None`` then library calculates and identifies the columns on its own.

- ord_dict(``dict``): Dictionary giving the column names as keys and their respective mapping dictionary as values.

- replace_cat_nulls(``int``): Integer value that is used to fill all null cells in categorical columns(taken from ``cat_cols`` parameter)

- drop_cols(``list``): List of column names to be dropped.

- fill_missing(``list``): List with the format [[columns that the method will be applied on], method]. For example: [["Age"],"mean"].

- fill_values(``dict``): Dictionary with keys as column names and values that fill the null records in corresponding column.

- one_hot(``bool``): ``True`` if one hot encoding desired. Default= ``False``.

- remove_outliers(``bool``): ``True`` if outlier records to be removed. Default= ``True``. If ``False`` then ``replace`` parameter must be given.

- replace(``int``): Integer value to replace all outliers with.

- first_quartile(``float``): Float value less than 1 and represents the first percentile marker. For more see :py:mod:`preprocessy.outliers`

- third_quartile(``float``): Float value less than 1 and represents the other percentile marker. For more see :py:mod:`preprocessy.outliers`

- type(``str``): The type of Scaler to be used. Options include "MinMaxScaler" | "BinaryScaler" | "StandardScaler".

- columns(``list``): List of columns in the dataframe.

- is_combined(``bool``): Parameter to determine whether columns should be scaled together as a group

- threshold(``dict``): Dictionary of threshold values where the key is the column name and the value is the threshold for that column.

- score_func(``callable``): Function taking two arrays X and y, and returning a pair of arrays
                     ``(scores, pvalues)`` or a single array with scores. ``score_func`` can be custom
                     or used from ``sklearn.feature_selection``

- k(``int``): Number of top features to select.

- test_size(``float,int``): Size of test set after splitting. Can take values from 0 - 1 for float point values, 0 - Number of samples for integer values. Is complementary to train size.

- train_size(``float,int``): Size of train set after splitting. Can take values from 0 - 1 for float point values, 0 - Number of samples for integer values. Is complementary to test size.

- n_splits(``int``): Number of folds to be made in K-fold cross validation. Must be at least 2. For more see :py:mod:`preprocessy.data_splitting.KFold`

- shuffle(``bool``): Decides whether to shuffle data before splitting.

- random_state(``int``): Seeding to be provided for shuffling before splitting.
