# dpp

Data Preprocessing library that provides customizable pipelines.

## Setup

* Clone the repo and install dependencies in a venv. `requirements_dev.txt` would automatically install `requirements.txt`

```bash
    $ pip install -r requirements_dev.txt
```

* Create a folder called `datasets` in the `root` directory. Its content can be found [here](https://drive.google.com/drive/folders/1qO3xrOVxSJDkNEcSZBoUPqq5cS_HazCZ?usp=sharing)

* All code goes inside `preprocessy`. All test scripts go inside `tests`. All evaluation scripts go in `evaluations`

## Steps before committing

* Run tests from `root` directory.

```bash
    $ python -m pytest
```

* Run linter from `root` directory.

```bash
    $ pylint *.py
```

* Run code formatter and spell checker from `root` directory

```bash
    $ black . && codespell --skip=".git,*.gif,*.png,*.PNG,./venv,*.json,./datasets,./.DS_Store,./tests/__pycache__"
```