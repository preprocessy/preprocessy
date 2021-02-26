import pytest
import numpy as np
import pandas as pd

from preprocessy.pipelines import Pipeline
from preprocessy.utils import num_of_samples
from preprocessy.exceptions import ArgumentsError


def read(params):
    params["df_copy"] = params["df"].copy()


def times_two(params):
    params["df"][params["col_1"]] *= 2


def squared(params):
    params["df"][params["col_2"]] **= 2


def split(params):
    n_samples = num_of_samples(params["df"])
    params["X_test"] = params["df"].iloc[: int(params["test_size"] * n_samples)]
    params["X_train"] = params["df"].iloc[int(params["test_size"] * n_samples) :]


class TestBasePipeline:
    def test_pipeline_arguments(self):

        with pytest.raises(ArgumentsError):
            Pipeline()

        with pytest.raises(ArgumentsError):
            Pipeline(steps=[read, times_two, squared, split])

        with pytest.raises(TypeError):
            Pipeline(steps=[read, "times_two", squared, split],params=["hello"])

        with pytest.raises(TypeError):
            Pipeline(steps=[read, times_two, squared, split], params=["hello"])

    def test_pipeline(self):

        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})

        params = {"df": df, "col_1": "A", "col_2": "B", "test_size": 0.2}

        pipeline = Pipeline(steps=[read, times_two, squared, split], params=params)
        pipeline.process()

        assert (
            pipeline.params["df"].loc[69, "A"]
            == pipeline.params["df_copy"].loc[69, "A"] * 2
        )
        assert (
            pipeline.params["df"].loc[42, "B"]
            == pipeline.params["df_copy"].loc[42, "B"] ** 2
        )

        assert len(pipeline.params["X_train"]) == 80

    def test_add(self):

        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})

        params = {"df": df, "col_1": "B", "test_size": 0.2}

        pipeline = Pipeline(steps=[read, times_two, split], params=params)
        pipeline.process()
        assert (
            pipeline.params["df"].loc[42, "B"]
            == pipeline.params["df_copy"].loc[42, "B"] * 2
        )

        pipeline.add(
            squared,
            {
                "col_2": "B",
            },
            index=1,
        )
        pipeline.process()
        num_0 = pipeline.params["df"].loc[42, "B"]
        num_1 = pipeline.params["df_copy"].loc[42, "B"]

        assert num_0 == (num_1 ** 2) * 2

    def test_remove(self):

        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})

        params = {"df": df, "col_1": "A", "col_2": "B", "test_size": 0.2}

        pipeline = Pipeline(steps=[read, times_two, squared, split], params=params)
        pipeline.process()

        assert len(pipeline.params["X_train"]) == 80

        pipeline.remove("split")

        pipeline.process()

        assert pipeline.params["df"].shape[0] == pipeline.params["df_copy"].shape[0]
