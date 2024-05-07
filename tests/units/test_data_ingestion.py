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


def test_split_always_returns_a_test_row(config):
    config.test_size = 0.0
    _, test = DataIngestion(config).split(pd.DataFrame({"a": range(5)}))
    assert len(test) == 1


def test_ingest_writes_all_three_files(config, source):
    train_path, test_path = DataIngestion(config).ingest(source)
    assert os.path.exists(config.raw_data_path)
    assert os.path.exists(train_path) and os.path.exists(test_path)


def test_ingested_files_can_be_read_back(config, source):
    train_path, test_path = DataIngestion(config).ingest(source)
    assert len(pd.read_csv(train_path)) + len(pd.read_csv(test_path)) == 10


def test_missing_source_file_raises(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        DataIngestion(config).ingest(str(tmp_path / "nope.csv"))
