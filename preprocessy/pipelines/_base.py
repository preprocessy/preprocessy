import warnings

import stringcase
from alive_progress import alive_bar
from colorama import Fore
from colorama import init
from colorama import Style
from prettytable import PrettyTable

from ..exceptions import ArgumentsError
from ..input import Reader
from .config import read_config

init()


class BasePipeline:
    def __init__(
        self,
        train_df_path=None,
        test_df_path=None,
        steps=None,
        config_file=None,
        params=None,
        custom_reader=None,
    ):
        self.params = params
        self.train_df_path = train_df_path
        self.test_df_path = test_df_path
        self.config_file = config_file
        self.steps = steps
        self.custom_reader = custom_reader
        self.__validate_input()

        if self.config_file and not self.params:
            self.params = read_config(self.config_file)

        if self.custom_reader is None:
            self.custom_reader = Reader().read_file

        self.add(
            self.custom_reader,
            {
                "train_df_path": self.train_df_path,
                "test_df_path": self.test_df_path,
            },
            index=0,
        )

    def __validate_input(self):

        if self.params and self.config_file:
            self.config_file = None
            warnings.warn(
                "'params' and 'config_file' both were provided. Using 'params'"
                " to construct the pipeline."
            )

        if not self.params and not self.config_file:
            raise ArgumentsError(
                "Both 'steps' and 'config_file' cannot be null. Please provide"
                " either a list of steps or path to a JSON config file."
            )

        if self.steps and not isinstance(self.steps, list):
            raise TypeError(
                f"'steps' should be of type 'list'. Received {self.steps} of"
                f" type {type(self.steps)}"
            )

        if self.steps:
            for step in self.steps:
                if not callable(step):
                    raise TypeError(
                        "All steps of the pipeline must be callable. Received"
                        f" {step} of type {type(step)}"
                    )

        if self.steps and not self.params and not self.config_file:
            raise ArgumentsError(
                "'params' dictionary or 'config_file' path to config file"
                " required for configuring pipeline. Received None"
            )

        if self.params and not isinstance(self.params, dict):
            raise TypeError(
                f"'params' should be of type dict. Received {self.params} of"
                f" type {type(self.params)}"
            )

        if self.config_file and not isinstance(self.config_file, str):
            raise TypeError(
                "'config_file' should be of type str. Received"
                f" {self.config_file} of type: {type(self.config_file)}"
            )

        if not self.train_df_path:
            raise ArgumentsError("'train_df_path' should not be None.")

        if not isinstance(self.train_df_path, str):
            raise TypeError(
                f"'train_df_path' should be of type str. Received {self.train_df_path} "
                f"of type {type(self.train_df_path)}"
            )

        if self.test_df_path:
            if not isinstance(self.test_df_path, str):
                raise TypeError(
                    f"'test_df_path' should be of type str. Received {self.test_df_path} "
                    f"of type {type(self.test_df_path)}"
                )

        if self.custom_reader and not callable(self.custom_reader):
            raise TypeError(
                "'custom_reader' should be a callable. Received"
                f" {self.custom_reader} of type {type(self.custom_reader)}"
            )

    def process(self):
        self.print_info()
        with alive_bar(
            len(self.steps),
            title="Pipeline Stages",
            enrich_print=False,
            force_tty=True,
        ) as bar:
            print("\nProcessing...\n")
            for step in self.steps:
                step(self.params)
                print(
                    f"==> Completed Stage: {stringcase.sentencecase(step.__name__)}\n"
                )
                bar()
        print(
            Fore.GREEN + "\nPipeline Completed Successfully\n" + Style.RESET_ALL
        )

    def __insert(self, index, func, params):
        self.steps.insert(index, func)
        for k, v in params.items():
            # TODO: Add warning if param with same name already exists
            self.params[k] = v

    def add(self, func=None, params=None, **kwargs):

        if not callable(func):
            raise TypeError(
                f"'func' should be a callable. Received {func} of type"
                f" {type(func)}"
            )

        if params and not isinstance(params, dict):
            raise TypeError(
                f"'params' should be of type dict. Received {params} of type"
                f" {type(params)}"
            )

        if "index" in kwargs.keys():
            self.__insert(kwargs.get("index"), func, params)
        elif "after" in kwargs.keys():
            index = -1
            for i, step in enumerate(self.steps):
                if step.__name__ == kwargs.get("after"):
                    index = i
                    break
            if index == -1:
                raise ValueError(
                    f"Function {kwargs.get('after')} is not a part of the"
                    " pipeline."
                )
            self.__insert(index + 1, func, params)
        elif "before" in kwargs.keys():
            index = -1
            for i, step in enumerate(self.steps):
                if step.__name__ == kwargs.get("before"):
                    index = i
                    break
            if index == -1:
                raise ValueError(
                    f"Function {kwargs.get('before')} is not a part of the"
                    " pipeline."
                )
            self.__insert(index, func, params)
        else:
            raise ArgumentsError(
                "No position was provided to insert the function into the"
                " pipeline"
            )

    def remove(self, func_name=None):
        if not isinstance(func_name, str):
            raise TypeError(
                f"'func_name' should be of type str. Received {func_name} of"
                f" type {type(func_name)}"
            )

        func = None
        for step in self.steps:
            if step.__name__ == func_name:
                func = step
                break

        if not func:
            raise ValueError(
                f"Function {func_name} is not a part of the pipeline."
            )

        self.steps.remove(func)

    def print_info(self):
        print(f"\nPipeline Class: {self.__class__.__name__}\n")
        table = PrettyTable(["Pipeline Property", "Value"])
        table.align = "l"
        table.add_row(["Train Dataframe Path", self.train_df_path])
        table.add_row(["Test Dataframe Path", self.test_df_path])
        table.add_row(["Config File Path", self.config_file])
        table.add_row(
            [
                "Pipeline Stages",
                ", ".join(
                    [
                        stringcase.sentencecase(stage.__name__)
                        for stage in self.steps
                    ]
                ),
            ]
        )
        table.add_row(["Total Pipeline Stages", len(self.steps)])
        table.add_row(["Total Params", len(self.params.keys())])
        print(table)
