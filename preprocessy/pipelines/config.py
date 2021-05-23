import copy
import json
import os.path
import warnings

content = dict()


def __validate_config(file_path):
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise FileNotFoundError(
            "Please make sure you to provide a valid config file that exists"
            f" at {file_path}"
        )


# return a dict/object of all the params read from the config
def read_config(file_path):
    global content
    __validate_config(file_path)

    try:
        with open(file_path) as f:
            content = json.load(f)
    except Exception as e:
        raise TypeError(
            f"Error occurred while reading the config file : {str(e)}"
        )

    if "train_df_path" in content:
        warnings.warn(
            "The dataset has to be passed as param to the Pipeline class, any"
            " value provided here will be overridden."
        )
    return content


# save the params object to a file
def save_config(file_path, params):
    try:
        with open(file_path, "w") as f:
            params_copy = copy.deepcopy(params)
            if "train_df" in params_copy.keys():
                params_copy.pop("train_df")
            if "test_df" in params_copy.keys():
                params_copy.pop("test_df")
            json.dump(params_copy, f, indent=2)
    except Exception as e:
        print(f"Error occurred while saving config to file : {str(e)}")
