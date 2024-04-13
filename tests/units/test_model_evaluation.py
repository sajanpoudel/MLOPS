import pytest

from src.components.model_evaluation import (
    evaluate,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def test_mae_of_a_perfect_prediction_is_zero():
    assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == 0


def test_mae_averages_the_absolute_errors():
    assert mean_absolute_error([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_rmse_penalises_large_errors_more():
    assert root_mean_squared_error([0, 0], [3, 4]) == pytest.approx(((9 + 16) / 2) ** 0.5)


def test_r2_is_one_for_a_perfect_fit():
    assert r2_score([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_r2_is_zero_when_predicting_the_mean():
    assert r2_score([1, 2, 3], [2, 2, 2]) == pytest.approx(0.0)


def test_r2_can_be_negative_for_bad_models():
    assert r2_score([1, 2, 3], [3, 2, 1]) < 0


def test_r2_of_constant_targets():
    assert r2_score([5, 5], [5, 5]) == 1.0
    assert r2_score([5, 5], [4, 6]) == 0.0
