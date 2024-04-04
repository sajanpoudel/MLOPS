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


def test_save_object_creates_missing_folders(tmp_path):
    path = tmp_path / "a" / "b" / "obj.pkl"
    save_object(str(path), [1, 2])
    assert path.exists()


def test_objects_with_nested_data_round_trip(tmp_path):
    data = {"scores": [0.1, 0.2], "meta": {"name": "model"}}
    path = str(tmp_path / "nested.pkl")
    save_object(path, data)
    assert load_object(path) == data


def test_save_object_failure_raises_custom_exception():
    with pytest.raises(CustomException):
        save_object("/proc/not-writable/obj.pkl", object())


def test_custom_exception_keeps_the_original_message():
    try:
        raise ValueError("bad value")
    except ValueError as e:
        assert "bad value" in str(CustomException(e, sys))
