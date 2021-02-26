import json
import os.path

content = dict()

def __validate_config(file_path):
    global content
    if not (os.path.exists(file_path) and os.path.isfile(file_path)):
        raise FileNotFoundError(f"Please make sure you provide a valid config file and the file exists at {file_path}")

    try:
        with open(file_path) as f:
            content = json.load(f)
    except Exception as e:
        raise TypeError(f"Error occurred while reading the config file : {str(e)}")

# return a dict/object of all the params read from the config
def read_config(file_path, steps):
    global content
    __validate_config(file_path)
    for step in steps:
        if step not in content.keys():
            raise ValueError(f"Expected parameters for {step}")
    return content

# save the params object to a file 
def save_config(file_path,params):
    try:
        with open(file_path,'w') as f:
            json.dump(params,f,indent=4)
    except Exception as e:
        print(f"Error occurred while saving config to file : {str(e)}")