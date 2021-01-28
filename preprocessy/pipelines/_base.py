import inspect
import warnings

from ..exceptions import ArgumentsError


class Pipeline:
    def __init__(self, steps=None, config_file=None, params=None):

        self.params = params
        self.config_file = config_file
        self.steps = steps
        self.output = {}
        self.__validate_input()

    def __validate_input(self):

        if self.steps and self.config_file:
            self.config_file = None
            warnings.warn(
                f"'steps' and 'config_file' both were provided. Using 'steps' to construct the pipeline."
            )

        if not self.steps and not self.config_file:
            raise ArgumentsError(
                f"Both 'steps' and 'config_file' cannot be null. Please provide either a list of steps or path to a JSON config file."
            )

        if self.steps and not isinstance(self.steps, list):
            raise TypeError(
                f"'steps' should be of type 'list'. Received {self.steps} of type {type(self.steps)}"
            )

        if self.steps:
            for step in self.steps:
                if not inspect.isfunction(step):
                    raise TypeError(
                        f"All steps of the pipeline must be functions. Received {step} of type {type(step)}"
                    )

        if self.steps and not self.params:
            raise ArgumentsError(
                f"'params' dictionary required for configuring pipeline. Received None"
            )

        if self.params and not isinstance(self.params, dict):
            raise TypeError(
                f"'params' should be of type dict. Received {self.params} of type {type(self.params)}"
            )

        if self.config_file and not isinstance(self.config_file, str):
            raise TypeError(
                f"'config_file' should be of type str. Received {self.config_file} of type: {type(self.config_file)}"
            )

    def process(self):
        for step in self.steps:
            step(self.params)

    def __insert(self, index, func, params):
        self.steps.insert(index, func)
        for k, v in params.items():
            self.params[k] = v

    def add(self, func=None, params=None, **kwargs):

        if not inspect.isfunction(func):
            raise TypeError(
                f"'steps' should be of type 'list'. Received {self.steps} of type {type(self.steps)}"
            )

        if params and not isinstance(params, dict):
            raise TypeError(
                f"'params' should be of type dict. Received {params} of type {type(params)}"
            )

        if "index" in kwargs.keys():
            self.__insert(kwargs.get("index"), func, params)
        elif "after" in kwargs.keys():
            index = -1
            for i, func in enumerate(self.steps):
                if func.__name__ == kwargs.get("after"):
                    index = i
                    break
            if index == -1:
                raise ValueError(
                    f"Function {kwargs.get('after')} is not a part of the pipeline."
                )
            self.__insert(index + 1, func, params)
        elif "before" in kwargs.keys():
            index = -1
            for i, func in enumerate(self.steps):
                if func.__name__ == kwargs.get("before"):
                    index = i
                    break
            if index == -1:
                raise ValueError(
                    f"Function {kwargs.get('before')} is not a part of the pipeline."
                )
            self.__insert(index - 1, func, params)
        else:
            raise ArgumentsError(
                f"No position was provided to insert the function into the pipeline"
            )

    def remove(self, func_name=None):
        if not isinstance(func_name, str):
            raise TypeError(
                f"'func_name' should be of type str. Received {func_name} of type {type(func_name)}"
            )

        func = None
        for step in self.steps:
            if step.__name__ == func_name:
                func = step
                break

        if not func:
            raise ValueError(f"Function {func_name} is not a part of the pipeline.")

        self.steps.remove(func)

    def info(self):
        # TODO: Formatting the output
        print(self.steps)
        print(self.params)
