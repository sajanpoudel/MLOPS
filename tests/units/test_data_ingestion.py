import os

import pandas as pd
import pytest

from src.components.data_ingestion import DataIngestion, DataIngestionConfig


@pytest.fixture
def config(tmp_path):
    return DataIngestionConfig(
        raw_data_path=str(tmp_path / "raw.csv"),
        train_data_path=str(tmp_path / "train.csv"),
        test_data_path=str(tmp_path / "test.csv"),
    )


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.csv"
    pd.DataFrame({"x": range(10), "y": range(10, 20)}).to_csv(path, index=False)
    return str(path)


def test_split_sizes_follow_test_size(config):
    train, test = DataIngestion(config).split(pd.DataFrame({"a": range(10)}))
    assert (len(train), len(test)) == (8, 2)


def test_split_keeps_every_row_once(config):
    train, test = DataIngestion(config).split(pd.DataFrame({"a": range(10)}))
    assert sorted(list(train["a"]) + list(test["a"])) == list(range(10))


def test_split_is_repeatable(config):
    frame = pd.DataFrame({"a": range(20)})
    first = DataIngestion(config).split(frame)
    second = DataIngestion(config).split(frame)
    assert list(first[1]["a"]) == list(second[1]["a"])
