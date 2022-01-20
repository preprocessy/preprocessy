
Parameters for pipeline configuration
=====================================

The parameters required by both the built-in pipelines as well as the custom pipelines
are given as a ``dict`` object or as a path to a ``json`` file. These params are used
by individual functions of the pipeline.

When writing custom functions, you can add custom params that are required by your function.
The params dict will be passed to your function when it is called as a part of the pipeline
and will have access to your custom parameter.

.. warning::
    If the name of your custom parameter conflicts with one of those required by the built-in
    functions, then the value will override the parameter and lead to uncertain outputs and errors.

    Please ensure that you do not use conflicting names for your custom parameters.

The paramters used by the built-in functions are listed below along with their meaning,
expected dtype and default value, if any. Any parameter marked ``*`` is required.

- **train_df_path \***

Path to the train dataset. Path should point to a file of one of the supported extensions.
For list of allowed extensions see :py:mod:`preprocessy.input`.

.. code:: python

    dtype: str
    example: "/Users/home/datasets/titanic.csv"

- **test_df_path**

Path to the test dataset. Path should point to a file of one of the supported extensions.
For list of allowed extensions see :py:mod:`preprocessy.input`.

.. code:: python

    dtype: str
    example: "/Users/home/datasets/titanic_test.csv"

- **target_label**

Name of the target column.

.. code:: python

    dtype: str
    example: "Survived"

- **cat_cols**

List of column names provided by user indicating these columns are to be encoded categorically. If ``None`` then ``Preprocessy`` analyses and identifies the columns on its own.

.. code:: python

    dtype: list[str]
    example: ["Gender", "Cabin", "Embarked"]

- **ord_dict**

Ordinal attributes require an associated weight mapping. This parameter is mapping from attribute column
name to weight mapping. Weight mapping is a dict specifying the weight associated with each unique value of
the attribute in consideration.

.. code:: python

    dtype: dict[str => dict[str => int | float]]
    example: {
        "Difficulty": {
            "Easy": 5
            "Medium": 10,
            "Hard": 15
        }
    }

- **replace_cat_nulls**

When handling missing data for categorical and ordinal attributes, the ``int`` provided here will be used
to replace the null value. If no value is provided, then the column will be dropped.

.. code:: python

    dtype: int
    example: 99

- **drop_cols**

List of column names to be dropped.

.. code:: python

    dtype: list[str]
    example: ["PassengerId", "Name"]

- **fill_missing**

Dictionary of format {"method": [col]} to indicate the method (``mean``/``median``) to be applied on specified list of columns.

.. code:: python

    dtype: dict["mean" | "median" => list[str]]
    example: {
        "mean": ["col_A", "col_B"],
        "median": ["col_C"]
    }

- **fill_values**

Dictionary with keys as column names and values that fill the null records in corresponding column.

.. code:: python

    dtype: dict[str => any]
    example: {
        "Age": 19,
        "Name": "John"
    }

- **one_hot**

``True`` if one hot encoding is desired. Default = ``False``.

.. code:: python

    dtype: bool

- **remove_outliers**

``True`` if outlier records are to be removed. If both ``remove_outliers`` and ``replace``
are ``False``, a warning will be raised and no operation will be performed. Default = ``True``

.. code:: python

    dtype: bool

- **replace**

Boolean value to indicate if the outliers need to be replaced by ``-999``. Default = ``False``.

.. code:: python

    dtype: bool

- **first_quartile**

Float value between 0 and 1, representing the first quartile marker. For more see :py:mod:`preprocessy.outliers`

.. code:: python

    dtype: float
    example: 0.25

- **third_quartile**

Float value between 0 and 1, representing the third quartile marker. For more see :py:mod:`preprocessy.outliers`

.. code:: python

    dtype: float
    example: 0.75

- **type**

The type of Scaler to be used. Default = ``StandardScaler``.

.. code:: python

    dtype: "MinMaxScaler" | "BinaryScaler" | "StandardScaler"
    example: "MinMaxScaler"

- **columns**

List of columns in the dataframe for which scaling is to be done. If ``None`` is provided, defaults to all columns of a Numeric dtype.

.. code:: python

    dtype: list[str]
    example : ["Fare"]

- **is_combined**

Parameter to determine whether columns should be scaled together as a group.

.. code:: python

    dtype: bool

- **threshold**

``BinaryScaler`` uses a dictionary of threshold values where the key is the column name and the
value is the threshold for that column. All values less than or equal to the threshold are scaled to 0.
Values above the threshold are scaled to 1. The default threshold value is 0.

.. code:: python

    dtype: dict[str => int | float]
    example: {
        "Age": 17
    }

- **score_func**

Function taking two arrays X and y, and returning a pair of arrays
``(scores, pvalues)`` or a single array with scores. ``score_func`` can be custom
or used from ``sklearn.feature_selection``

.. code:: python

    dtype: func(iterable, iterable) => (list[float], list[float])
    example: f_classif from sklearn

- **k**

Number of top features to select.

.. code:: python

    dtype: int
    example: 10

- **test_size**

Size of test set after splitting. Can take values from 0 - 1 for floating point values,
0 - Number of samples for integer values. It is complementary to train size.

.. code:: python

    dtype: int | float
    example: 0.2, 200

- **train_size**

Size of train set after splitting. Can take values from 0 - 1 for floating point values,
0 - Number of samples for integer values. It is complementary to test size. If both ``train_size``
and ``test_size`` are given, then ``train_size + test_size`` should be equal to 1 if sizes are
floating point values, else the total size of the dataset.

.. code:: python

    dtype: int | float
    example: 0.8, 800

- **n_splits**

Number of folds to be made in K-fold cross validation. Must be at least 2.
For more see :py:mod:`preprocessy.data_splitting.KFold`

.. code:: python

    dtype: int
    example: 5

- **shuffle**

Decides whether to shuffle data before splitting. Default = ``False``

.. code:: python

    dtype: bool

- **random_state**

Seeding to be provided for shuffling before splitting. Requires ``shuffle`` to be ``True``.

.. code:: python

    dtype: int
    example: 0
