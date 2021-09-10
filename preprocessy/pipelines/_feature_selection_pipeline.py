from preprocessy.data_splitting import Split
from preprocessy.encoding import Encoder
from preprocessy.feature_selection import SelectKBest
from preprocessy.missing_data import NullValuesHandler
from preprocessy.outliers import HandleOutlier
from preprocessy.parse import Parser
from preprocessy.scaling import Scaler

from ._base import BasePipeline


class FeatureSelectionPipeline(BasePipeline):
    """
    Pre-built pipeline that can be used for feature selection

    The steps of the pipeline are:

    1. Parser
    2. NullValuesHandler
    3. Encoder
    4. HandleOutlier
    5. Scaler
    6. SelectKBest
    7. Split

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
        steps = [
            Parser().parse_dataset,
            NullValuesHandler().execute,
            Encoder().encode,
            HandleOutlier().handle_outliers,
            Scaler().execute,
            SelectKBest().fit_transform,
            Split().train_test_split,
        ]
        super().__init__(
            train_df_path=train_df_path,
            test_df_path=test_df_path,
            steps=steps,
            config_file=config_file,
            params=params,
            custom_reader=custom_reader,
        )
