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
from .config import save_config

init()


class BasePipeline:

    """The ``BasePipeline`` Class can be used to create your own customized pipeline.

    :param train_df_path: Path to train dataframe
              Should not be ``None``
    :type train_df_path: str

    :param test_df_path: Path to train dataframe
              Should not be ``None``
    :type test_df_path: str

    :param steps: A list of functions which will be executed sequentially.
            All the functions should be callables
    :type steps: list

    :param params: A dictionary containing the parameters that are needed for configuring the pipeline
    :type params: dict

    :param config_file: Path to a config file that consists the parameters for configuring the pipeline. An alternative to ``params``. A config file for the current ``params`` dictionary can be generated using the ``save_config`` utility
    :type config_file: str

    :param custom_reader: Custom function to read the data
    :type custom_reader: callable

    """

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
        self.config_drop_keys = [
            "train_df",
            "test_df",
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "train_df_path",
            "test_df_path",
        ]
        if steps is None:
            self.steps = []
        else:
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

        if not isinstance(self.steps, list):
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
        """Method that executes the pipeline sequentially."""
        self.print_info()
        with alive_bar(
            len(self.steps),
            title="Pipeline Stages",
            enrich_print=False,
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
        if params:
            for k, v in params.items():
                if k in self.params:
                    raise ValueError(
                        f"Non unique parameter name. Param with name {k} already exists."
                    )
                self.params[k] = v

    def add(self, func=None, params=None, **kwargs):

        """Method to add another function to the pipeline after it has been constructed

        :param func: The function to be added
        :type func: callable

        :param params: Dictionary of configurable parameters to be added to the existing
                    ``params`` dictionary. Can be empty or ``None``.
        :type params: dict

        :param index: The index at which the function is to be inserted.
        :type index: int

        :param after: The step name after which the function should be added
        :type after: str

        :param before: The step name before which the function should be added
        :type before: str

        To add a function, either the index position or the ``before``/``after`` positional arguments can be supplied

        If ``index``, ``after`` and ``before`` are all provided, the method will follow the priority: ``index`` > ``after`` > ``before``

        :raises ArgumentsError: If no position is provided to insert the function into the pipeline

        :raises ValueError: If ``params`` contains a key that already exists in ``self.params``

        """
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
        """Method to remove a function from the pipeline

        :param func_name: The name of the function which has to be removed from the pipeline
        :type func_name: str

        :raises TypeError: If ``func_name`` is not of type ``str``

        """
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

    def save_config(self, file_path, config_drop_keys=None):
        """Method to save the ``params`` to a ``JSON`` config file

        :param file_path: Path where the config file must be created
        :type file_path: str
        :param config_drop_keys: List of param keys that must not be stored in the config file,
                            defaults to ``["train_df", "test_df", "X_train", "X_test", "y_train", "y_test"]``
        :type config_drop_keys: list, optional
        """
        if not config_drop_keys:
            config_drop_keys = self.config_drop_keys
        save_config(file_path, self.params, config_drop_keys)

    def print_info(self):
        """Prints the current configuration of the pipeline. Shows the steps, dataframe paths and config paths."""
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
