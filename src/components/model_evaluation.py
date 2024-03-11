"""Metrics used to compare regression models."""

import math


def mean_absolute_error(actual, predicted):
    """Average absolute difference between actual and predicted values."""
    _check_lengths(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
