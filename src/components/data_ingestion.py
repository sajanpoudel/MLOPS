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


class DataIngestion:
    """Loads the raw data and writes the train and test splits."""

    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    def split(self, frame: pd.DataFrame):
        """Shuffle the rows and cut them into (train, test)."""
        shuffled = frame.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
        test_rows = max(1, int(round(len(shuffled) * self.config.test_size)))
        return shuffled.iloc[test_rows:].reset_index(drop=True), shuffled.iloc[:test_rows]
