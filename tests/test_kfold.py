import numpy as np
import pytest

from preprocessy.resampling import KFold


@pytest.mark.parametrize(
    "test_input1, test_input2, test_input3",
    [
        (2.3, False, None),
        (0, False, None),
        (5, 1, None),
        (5, True, 4.5),
        (5, False, 69),
    ],
)
def test_kfold(test_input1, test_input2, test_input3):
    with pytest.raises(ValueError):
        KFold(
            n_splits=test_input1, shuffle=test_input2, random_state=test_input3
        )


def test_split():
    with pytest.raises(ValueError):
        arr = np.arange(10)
        kFold = KFold(n_splits=20)
        for train_indices, test_indices in kFold.split(arr):
            print(
                f"Train indices: {train_indices}\nTest indices:"
                f" {test_indices}"
            )

    arr = np.arange(12)
    kFold = KFold(n_splits=3, shuffle=True, random_state=69)
    for train_indices, test_indices in kFold.split(arr):
        assert len(train_indices) == 8
        assert len(test_indices) == 4
