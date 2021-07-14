from ._base import Pipeline
from preprocessy.encoding import Encoder
from preprocessy.handlenullvalues import NullValuesHandler
from preprocessy.outliers import HandleOutlier
from preprocessy.parse import Parser
from preprocessy.resampling import Split
from preprocessy.scaling import Scaler


class Preprocessy(Pipeline):
    def __init__(
        self,
        train_df_path,
        test_df_path=None,
        config_file=None,
        params=None,
        custom_reader=None,
    ):
        print("----------Initializing pipeline----------")
        steps = [
            Parser().parse_dataset,
            NullValuesHandler().execute,
            Encoder().encode,
            Scaler().execute,
            HandleOutlier().handle_outliers,
            Split().train_test_split,
        ]
        super().__init__(
            train_df_path=train_df_path,
            test_df_path=test_df_path,
            steps=steps,
            config_file=config_file,
            params=params,
        )
