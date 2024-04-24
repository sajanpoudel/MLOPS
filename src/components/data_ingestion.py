"""Reads a csv file and splits it into train and test files."""

import os
from dataclasses import dataclass

import pandas as pd


@dataclass
class DataIngestionConfig:
    """Where the files are read from and written to."""

    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    test_size: float = 0.2
    random_state: int = 42
