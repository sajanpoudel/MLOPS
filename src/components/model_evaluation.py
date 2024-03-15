"""Metrics used to compare regression models."""

import math


def mean_absolute_error(actual, predicted):
    """Average absolute difference between actual and predicted values."""
    _check_lengths(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def _check_lengths(actual, predicted):
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if len(actual) == 0:
        raise ValueError("at least one value is needed")


def root_mean_squared_error(actual, predicted):
    """Square root of the average squared difference."""
    _check_lengths(actual, predicted)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def r2_score(actual, predicted):
    """Share of the variance of actual that the predictions explain (1.0 is perfect)."""
    _check_lengths(actual, predicted)
    mean = sum(actual) / len(actual)
    total = sum((a - mean) ** 2 for a in actual)
    if total == 0:
        return 1.0 if all(a == p for a, p in zip(actual, predicted)) else 0.0
    residual = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1 - residual / total
