import pytest

from src.components.model_evaluation import (
    evaluate,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def test_mae_of_a_perfect_prediction_is_zero():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0
