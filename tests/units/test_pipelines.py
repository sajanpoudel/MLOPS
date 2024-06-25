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


def test_training_reports_the_metrics_and_the_model(trained):
    metrics, _ = trained
    assert set(metrics) == {"mae", "rmse", "r2", "model"}


def test_the_trained_model_fits_the_data_well(trained):
    metrics, _ = trained
    assert metrics["r2"] > 0.9


def test_predict_returns_one_value_per_row(trained):
    _, model_path = trained
    rows = [{"a": 0.5, "b": 0.5, "kind": "x"}, {"a": 0.1, "b": 0.9, "kind": "y"}]
    assert len(predict(model_path, rows)) == 2


def test_predictions_follow_the_pattern_in_the_data(trained):
    _, model_path = trained
    low, high = predict(model_path, [{"a": 0.0, "b": 0.0, "kind": "y"}, {"a": 1.0, "b": 1.0, "kind": "x"}])
    assert high > low


def test_predict_without_a_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict(str(tmp_path / "missing.pkl"), [{"a": 1}])
