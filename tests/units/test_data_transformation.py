import numpy as np
import pandas as pd
import pytest

from src.components.data_transformation import DataTransformation, DataTransformationConfig


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "size": [1.0, 2.0, np.nan, 4.0],
            "city": ["a", "b", "a", None],
            "target": [10, 20, 30, 40],
        }
    )
