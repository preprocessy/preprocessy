import json
import os.path

class Config:

    def __init__(self, configPath=None, stepsStr=[]):
        self.filePath=configPath
        self.steps=stepsStr
        self.content=dict()
        self.__validateConfig()

    # check if file exists and contains valid JSON
    def __validateConfig(self):
        if not (os.path.exists(self.filePath) and os.path.isfile(self.filePath)):
            print(f"Please make sure you provide a valid config file and the file exists at {self.filePath}")
            raise FileNotFoundError

        try:
            with open(self.filePath) as f:
                self.content = json.load(f)
        except Exception as e:
            print(f"Error occurred while reading the config file : {str(e)}")
            raise TypeError

    # return a dict/object of all the params read from the config
    def readConfig(self):
        for step in self.steps:
            if step not in self.content.keys():
                print(f"Expected parameters for {step}")
                raise ValueError
        return self.content