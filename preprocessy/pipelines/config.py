import json
import os.path
import warnings

content = dict()


def __validate_config(file_path):
    global content
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise FileNotFoundError(
            f"Please make sure you to provide a valid config file that exists at {file_path}"
        )

    try:
        with open(file_path) as f:
            content = json.load(f)
    except Exception as e:
        raise TypeError(f"Error occurred while reading the config file : {str(e)}")


# return a dict/object of all the params read from the config
def read_config(file_path):
    global content
    __validate_config(file_path)
    if "df" in content:
        warnings.warn(
            f"The dataset has to be passed as param to the Pipeline class, any value provided here will be overridden."
        )
        if not (os.path.exists(content["df"]) and os.path.isfile(content["df"])):
            raise FileNotFoundError(f"No dataset file found at {content['df']}")
    return content


# save the params object to a file
def save_config(file_path, params):
    try:
        with open(file_path, "w") as f:
            json.dump(params, f, indent=2)
    except Exception as e:
        print(f"Error occurred while saving config to file : {str(e)}")
