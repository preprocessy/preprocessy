import numpy as np
import pandas as pd
import pytest
from preprocessy.exceptions import ArgumentsError
from preprocessy.pipelines import BasePipeline
from preprocessy.utils import num_of_samples


def custom_read(params):
    params["train_df"] = pd.read_csv(params["train_df_path"])
    if params["test_df_path"]:
        params["test_df"] = pd.read_csv(params["test_df_path"])
        params["test_df_copy"] = params["test_df"].copy()
    params["train_df_copy"] = params["train_df"].copy()


def times_two(params):
    params["train_df"][params["col_1"]] *= 2


def squared(params):
    params["train_df"][params["col_2"]] **= 2


def split(params):
    n_samples = num_of_samples(params["train_df"])
    params["X_test"] = params["train_df"].iloc[
        : int(params["test_size"] * n_samples)
    ]
    params["X_train"] = params["train_df"].iloc[
        int(params["test_size"] * n_samples) :
    ]


@pytest.mark.parametrize(
    "error, train_df_path, steps, config_file, params, custom_reader",
    [
        (ArgumentsError, None, None, None, None, None),
        (
            ArgumentsError,
            None,
            [custom_read, times_two, squared, split],
            None,
            None,
            None,
        ),
        (
            TypeError,
            "./datasets/configs/dataset.csv",
            [custom_read, "times_two", squared, split],
            None,
            ["hello"],
            None,
        ),
        (
            TypeError,
            "./datasets/configs/dataset.csv",
            [custom_read, times_two, squared, split],
            None,
            ["hello"],
            None,
        ),
        (
            TypeError,
            "./datasets/configs/dataset.csv",
            [times_two, squared, split],
            None,
            {"col_1": "A"},
            "custom_read",
        ),
    ],
)
def test_pipeline_arguments(
    error, train_df_path, steps, config_file, params, custom_reader
):

    with pytest.raises(error):
        BasePipeline(
            train_df_path=train_df_path,
            steps=steps,
            config_file=config_file,
            params=params,
            custom_reader=custom_reader,
        )


def test_pipeline_with_default_reader():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)

    params = {
        "col_1": "A",
        "col_2": "B",
        "test_size": 0.2,
    }

    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, squared, split],
        params=params,
    )
    pipeline.process()

    assert "train_df" in pipeline.params.keys()


def test_pipeline_with_custom_reader():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)

    params = {
        "col_1": "A",
        "col_2": "B",
        "test_size": 0.2,
    }

    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, squared, split],
        params=params,
        custom_reader=custom_read,
    )
    pipeline.process()

    assert (
        pipeline.params["train_df"].loc[69, "A"]
        == pipeline.params["train_df_copy"].loc[69, "A"] * 2
    )
    assert (
        pipeline.params["train_df"].loc[42, "B"]
        == pipeline.params["train_df_copy"].loc[42, "B"] ** 2
    )

    assert len(pipeline.params["X_train"]) == 80


def test_add():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)
    params = {
        "col_1": "A",
        "test_size": 0.2,
    }
    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, split],
        params=params,
    )
    pipeline.process()
    assert pipeline.params["train_df"].loc[42, "A"] == df.loc[42, "A"] * 2
    pipeline.add(
        squared,
        {
            "col_2": "A",
        },
        before="times_two",
    )
    pipeline.process()
    num_0 = pipeline.params["train_df"].loc[42, "A"]
    num_1 = df.loc[42, "A"]
    assert num_0 == (num_1 ** 2) * 2
    pipeline.remove("squared")
    pipeline.add(squared, {"col_2": "A"}, after="read_file")
    pipeline.process()
    num_0 = pipeline.params["train_df"].loc[42, "A"]
    num_1 = df.loc[42, "A"]
    assert num_0 == (num_1 ** 2) * 2


def test_add_without_params():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)
    params = {
        "col_1": "A",
        "col_2": "B",
        "test_size": 0.2,
    }
    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, split],
        params=params,
    )
    pipeline.add(
        squared,
        before="times_two",
    )
    pipeline.process()
    assert pipeline.params["train_df"].loc[42, "B"] == df.loc[42, "B"] ** 2


def test_duplicate_param():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)
    params = {
        "col_1": "A",
        "col_2": "B",
        "test_size": 0.2,
    }
    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, split],
        params=params,
    )
    with pytest.raises(ValueError):
        pipeline.add(
            squared,
            {
                "col_2": "A",
            },
            before="times_two",
        )


def test_remove():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)
    params = {
        "col_1": "A",
        "col_2": "B",
        "test_size": 0.2,
    }
    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, squared, split],
        params=params,
    )
    pipeline.process()
    assert len(pipeline.params["X_train"]) == 80
    pipeline.remove("split")
    pipeline.process()
    assert pipeline.params["train_df"].shape[0] == df.shape[0]


def test_config():
    df = pd.DataFrame({"A": np.arange(1, 100), "B": np.arange(1, 100)})
    _ = df.to_csv("./datasets/configs/dataset.csv", index=False)
    params = {"col_1": "A", "col_2": "B", "test_size": 0.2, "X_train": df}
    config_path = "./datasets/configs/pipeline_config.json"
    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, squared, split],
        params=params,
        custom_reader=custom_read,
    )
    pipeline.process()

    # Drop custom key
    pipeline.config_drop_keys.append("train_df_copy")

    pipeline.save_config(config_path)

    pipeline = BasePipeline(
        train_df_path="./datasets/configs/dataset.csv",
        steps=[times_two, squared, split],
        config_file=config_path,
        custom_reader=custom_read,
    )

    assert "X_train" not in pipeline.params
    assert "train_df_copy" not in pipeline.params
