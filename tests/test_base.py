import pytest
import numpy as np
import pandas as pd

from preprocessy.pipelines import Pipeline
from preprocessy.utils import num_of_samples
from preprocessy.exceptions import ArgumentsError
from preprocessy.pipelines.config import save_config


def read(params):
    params["read"]["df"]=pd.DataFrame.from_dict(params["read"]["df"])
    params["df_copy"] = params["read"]["df"].copy()


def times_two(params):
    params["read"]["df"][params["times_two"]["col_1"]] *= 2


def squared(params):
    params["read"]["df"][params["squared"]["col_2"]] **= 2


def split(params):
    n_samples = num_of_samples(params["read"]["df"])
    params["X_test"] = params["read"]["df"].iloc[: int(params["split"]["test_size"] * n_samples)]
    params["X_train"] = params["read"]["df"].iloc[int(params["split"]["test_size"] * n_samples) :]


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

        params = {"read":{"df": df.to_dict()}, "times_two":{"col_1": "A"}, "squared":{"col_2": "B"},"split": {"test_size": 0.2}}

        pipeline = Pipeline(steps=[read, times_two, squared, split], params=params)
        pipeline.process()

        assert (
            pipeline.params["read"]["df"].loc[69, "A"]
            == pipeline.params["df_copy"].loc[69, "A"] * 2
        )
        assert (
            pipeline.params["read"]["df"].loc[42, "B"]
            == pipeline.params["df_copy"].loc[42, "B"] ** 2
        )

        assert len(pipeline.params["X_train"]) == 80

    def test_add(self):

        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})

        params = {"read":{"df": df.to_dict()}, "times_two":{"col_1": "A"},"split": {"test_size": 0.2}}

        pipeline = Pipeline(steps=[read, times_two, split], params=params)
        pipeline.process()
        assert (
            pipeline.params["read"]["df"].loc[42, "A"]
            == pipeline.params["df_copy"].loc[42, "A"] * 2
        )

        pipeline.add(
            squared,
            {"squared":{
                "col_2": "B",
            }},
            index=1,
        )
        pipeline.process()
        num_0 = pipeline.params["read"]["df"].loc[42, "B"]
        num_1 = pipeline.params["df_copy"].loc[42, "B"]

        assert num_0 == (num_1 ** 2)

    def test_remove(self):

        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})

        params = {"read":{"df": df.to_dict()}, "times_two":{"col_1": "A"}, "squared":{"col_2": "B"},"split": {"test_size": 0.2}}

        pipeline = Pipeline(steps=[read, times_two, squared, split], params=params)
        pipeline.process()

        assert len(pipeline.params["X_train"]) == 80

        pipeline.remove("split")

        pipeline.process()

        assert pipeline.params["read"]["df"].shape[0] == pipeline.params["df_copy"].shape[0]

    def test_config(self):
        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})

        params = {"read":{"df": df.to_dict()}, "times_two":{"col_1": "A"}, "squared":{"col_2": "B"},"split": {"test_size": 0.2}}
        config_path = "./datasets/configs/pipeline_config.json" 
        save_config(config_path, params)

        pipeline = Pipeline(steps=[read, times_two, squared, split], config_file=config_path)
        pipeline.process()

        assert len(pipeline.params["X_train"]) == 80

        pipeline.remove("split")

        pipeline.process()

        assert pipeline.params["read"]["df"].shape[0] == pipeline.params["df_copy"].shape[0]
