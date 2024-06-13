"""Runs ingestion, transformation, training and evaluation in order."""

import joblib
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import evaluate
from src.components.model_trainer import train_best_model


def run_training(source_csv: str, target_column: str, model_path: str, ingestion=None):
    """Train on source_csv and save {"preprocessor", "model"} to model_path. Returns the metrics."""
    from src.components.data_transformation import DataTransformationConfig

    ingestion = ingestion or DataIngestion()
    train_path, test_path = ingestion.ingest(source_csv)
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    transformer = DataTransformation(DataTransformationConfig(target_column))
    x_train, y_train, x_test, y_test, preprocessor = transformer.fit_transform(train, test)
    name, model, scores = train_best_model(x_train, y_train, x_test, y_test)
    joblib.dump({"preprocessor": preprocessor, "model": model, "name": name}, model_path)
    metrics = evaluate(list(y_test), list(model.predict(x_test)))
    metrics["model"] = name
    return metrics
