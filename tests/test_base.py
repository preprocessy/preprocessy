import pytest
import numpy as np
import pandas as pd

from preprocessy.pipelines import Pipeline
from preprocessy.utils import num_of_samples
from preprocessy.exceptions import ArgumentsError
from preprocessy.pipelines.config import save_config


def custom_read(params):
    params["df"] = pd.read_csv(params["df_path"])
    params["df_copy"] = params["df"].copy()


def times_two(params):
    params["df"][params["col_1"]] *= 2


def squared(params):
    params["df"][params["col_2"]] **= 2


def split(params):
    n_samples = num_of_samples(params["df"])
    params["X_test"] = params["df"].iloc[
        : int(params["test_size"] * n_samples)
    ]
    params["X_train"] = params["df"].iloc[
        int(params["test_size"] * n_samples) :
    ]


class TestBasePipeline:
    def test_pipeline_arguments(self):

        with pytest.raises(ArgumentsError):
            Pipeline()

        with pytest.raises(ArgumentsError):
            Pipeline(steps=[custom_read, times_two, squared, split])

        with pytest.raises(TypeError):
            Pipeline(
                df_path="./datasets/configs/dataset.csv",
                steps=[custom_read, "times_two", squared, split],
                params=["hello"],
            )

        with pytest.raises(TypeError):
            Pipeline(
                df_path="./datasets/configs/dataset.csv",
                steps=[custom_read, times_two, squared, split],
                params=["hello"],
            )

        with pytest.raises(TypeError):
            Pipeline(
                df_path="./datasets/configs/dataset.csv",
                steps=[times_two, squared, split],
                params={"col_1": "A"},
                custom_reader="custom_read",
            )

    def test_pipeline_with_default_reader(self):
        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
        df_path = df.to_csv("./datasets/configs/dataset.csv", index=False)

        params = {
            "col_1": "A",
            "col_2": "B",
            "test_size": 0.2,
        }

        pipeline = Pipeline(
            df_path="./datasets/configs/dataset.csv",
            steps=[times_two, squared, split],
            params=params,
        )
        pipeline.process()

        assert "df" in pipeline.params.keys()
        assert "summary" in pipeline.params.keys()
        assert "stats" in pipeline.params.keys()

    def test_pipeline_with_custom_reader(self):

        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
        df_path = df.to_csv("./datasets/configs/dataset.csv", index=False)

        params = {
            "col_1": "A",
            "col_2": "B",
            "test_size": 0.2,
            "df": "./datasets/configs/dataset.csv",
        }

        pipeline = Pipeline(
            df_path="./datasets/configs/dataset.csv",
            steps=[times_two, squared, split],
            params=params,
            custom_reader=custom_read,
        )
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
        df_path = df.to_csv("./datasets/configs/dataset.csv", index=False)
        params = {
            "col_1": "A",
            "test_size": 0.2,
        }
        pipeline = Pipeline(
            df_path="./datasets/configs/dataset.csv",
            steps=[times_two, split],
            params=params,
        )
        pipeline.process()
        assert pipeline.params["df"].loc[42, "A"] == df.loc[42, "A"] * 2
        pipeline.add(
            squared,
            {
                "col_2": "A",
            },
            before="times_two",
        )
        pipeline.process()
        num_0 = pipeline.params["df"].loc[42, "A"]
        num_1 = df.loc[42, "A"]
        assert num_0 == (num_1 ** 2) * 2
        pipeline.remove("squared")
        pipeline.add(squared, {"col_2": "A"}, after="read_file")
        pipeline.process()
        num_0 = pipeline.params["df"].loc[42, "A"]
        num_1 = df.loc[42, "A"]
        assert num_0 == (num_1 ** 2) * 2

    def test_remove(self):
        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
        df_path = df.to_csv("./datasets/configs/dataset.csv", index=False)
        params = {
            "col_1": "A",
            "col_2": "B",
            "test_size": 0.2,
        }
        pipeline = Pipeline(
            df_path="./datasets/configs/dataset.csv",
            steps=[times_two, squared, split],
            params=params,
        )
        pipeline.process()
        assert len(pipeline.params["X_train"]) == 80
        pipeline.remove("split")
        pipeline.process()
        assert pipeline.params["df"].shape[0] == df.shape[0]

    def test_config(self):
        df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
        df_path = df.to_csv("./datasets/configs/dataset.csv", index=False)
        params = {
            "df": "./datasets/configs/dataset.csv",
            "col_1": "A",
            "col_2": "B",
            "test_size": 0.2,
        }
        config_path = "./datasets/configs/pipeline_config.json"
        save_config(config_path, params)

        pipeline = Pipeline(
            df_path="./datasets/configs/dataset.csv",
            steps=[times_two, squared, split],
            config_file=config_path,
            custom_reader=custom_read,
        )
        pipeline.process()
        assert len(pipeline.params["X_train"]) == 80
        pipeline.remove("split")
        pipeline.process()
        assert (
            pipeline.params["df"].shape[0]
            == pipeline.params["df_copy"].shape[0]
        )
