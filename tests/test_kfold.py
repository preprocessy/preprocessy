import numpy as np
import pandas as pd
import pytest
from preprocessy.resampling import KFold

class TestKFold:

    def test_kfold(self):
        with pytest.raises(ValueError):
            KFold(n_splits=2.3)
        
        with pytest.raises(ValueError):
            KFold(n_splits=0)

        with pytest.raises(ValueError):
            KFold(shuffle=1)
        
        with pytest.raises(ValueError):
            KFold(shuffle=True,random_state=4.5)

        with pytest.raises(ValueError):
            KFold(shuffle=False,random_state=69)

    def test_split(self):

        with pytest.raises(ValueError):
            arr = np.arange(10)
            kFold = KFold(n_splits=20)
            for train_indices, test_indices in kFold.split(arr):
                print(f"Train indices: {train_indices}\nTest indices: {test_indices}")

        arr = np.arange(12)
        kFold = KFold(n_splits=3,shuffle=True,random_state=69)
        for train_indices, test_indices in kFold.split(arr):
            assert len(train_indices) == 8
            assert len(test_indices) == 4
