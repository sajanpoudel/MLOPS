import numpy as np
import pandas as pd
import pytest

from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.pipeline.prediction_pipeline import predict
from src.pipeline.training_pipeline import run_training


@pytest.fixture
def trained(tmp_path):
    rng = np.random.RandomState(1)
    frame = pd.DataFrame({"a": rng.rand(80), "b": rng.rand(80), "kind": rng.choice(["x", "y"], 80)})
    frame["price"] = 4 * frame["a"] + 2 * frame["b"] + (frame["kind"] == "x") * 1.5
    source = tmp_path / "data.csv"
    frame.to_csv(source, index=False)
    config = DataIngestionConfig(
        raw_data_path=str(tmp_path / "raw.csv"),
        train_data_path=str(tmp_path / "train.csv"),
        test_data_path=str(tmp_path / "test.csv"),
    )
    model_path = str(tmp_path / "model.pkl")
    metrics = run_training(str(source), "price", model_path, ingestion=DataIngestion(config))
    return metrics, model_path
