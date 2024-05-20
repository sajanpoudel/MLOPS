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


def test_split_columns_separates_numbers_from_text(frame):
    numeric, categorical = DataTransformation().split_columns(frame)
    assert numeric == ["size"]
    assert categorical == ["city"]


def test_split_columns_ignores_the_target(frame):
    numeric, _ = DataTransformation().split_columns(frame)
    assert "target" not in numeric


def test_custom_target_column(frame):
    renamed = frame.rename(columns={"target": "price"})
    numeric, _ = DataTransformation(DataTransformationConfig("price")).split_columns(renamed)
    assert numeric == ["size"]
