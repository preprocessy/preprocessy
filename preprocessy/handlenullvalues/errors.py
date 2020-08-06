class Error(Exception):
    pass


class ArgumentsError(Error):
    def __init__(self, message):
        self.message = message
