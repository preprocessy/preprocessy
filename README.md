# Preprocessy

[![Build Status](https://travis-ci.org/preprocessy/preprocessy.svg?branch=master)](https://travis-ci.org/preprocessy/preprocessy)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-sucess.svg)](https://gitHub.com/preprocessy/preprocessy/graphs/commit-activity)
[![Issues Open](https://img.shields.io/github/issues/preprocessy/preprocessy)](https://github.com/preprocessy/preprocessy/issues)
[![Forks](https://img.shields.io/github/forks/preprocessy/preprocessy)](https://github.com/preprocessy/preprocessy/forks)
[![Stars](https://img.shields.io/github/stars/preprocessy/preprocessy)](https://github.com/preprocessy/preprocessy/stars)
[![GitHub contributors](https://img.shields.io/github/contributors/preprocessy/preprocessy)](https://gitHub.com/preprocessy/preprocessy/graphs/contributors/)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/dwyl/esta/issues)
[![MIT license](https://img.shields.io/badge/License-MIT-informational.svg)](https://lbesson.mit-license.org/)

Preprocessy is a library that provides data preprocessing pipelines for machine learning. It bundles all the common preprocessing steps that are performed on the data to prepare it for machine learning models. It aims to do so in a manner that is independent of the source and type of dataset. Hence, it provides a set of functions that have been generalized to different types of data. 

The pipelines themselves are composed of these functions and flexible so that the users can customise them by adding their processing functions or removing pipeline functions according to their needs. The pipelines thus provide an abstract and high-level interface to the users.


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
    $ pytest -v -s
```

* Run code formatter and spell checker from `root` directory

```bash
    $ black . && codespell --skip=".git,*.gif,*.png,*.PNG,./venv,*.json,./datasets,.DS_Store,*.pyc"
```
