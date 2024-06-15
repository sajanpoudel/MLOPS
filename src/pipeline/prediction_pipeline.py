"""Loads a saved model and predicts for new rows."""

import joblib
import pandas as pd


def predict(model_path: str, rows: list) -> list:
    """Predict a value for every row (a list of dictionaries with the feature columns)."""
    saved = joblib.load(model_path)
    features = saved["preprocessor"].transform(pd.DataFrame(rows))
    return [float(value) for value in saved["model"].predict(features)]
