import sys

import pytest

from src.exception.exception import CustomException
from src.utils.utils import load_object, save_object


def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "models" / "model.pkl")
    save_object(path, {"weights": [1, 2, 3]})
    assert load_object(path) == {"weights": [1, 2, 3]}


def test_load_missing_file_raises_custom_exception(tmp_path):
    with pytest.raises(CustomException) as error:
        load_object(str(tmp_path / "missing.pkl"))
    assert "missing.pkl" in str(error.value)


def test_custom_exception_reports_line_number():
    try:
        1 / 0
    except ZeroDivisionError as e:
        message = str(CustomException(e, sys))
    assert "division by zero" in message
    assert "line" in message
