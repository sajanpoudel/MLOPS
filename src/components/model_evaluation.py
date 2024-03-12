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
