"""Builds the preprocessing that turns raw columns into model features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    """Which column is predicted."""

    target_column: str = "target"


class DataTransformation:
    """Imputes, scales and encodes the columns of a data frame."""

    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()

    def split_columns(self, frame: pd.DataFrame):
        """Return (numeric, categorical) feature column names, without the target."""
        features = frame.drop(columns=[self.config.target_column], errors="ignore")
        numeric = list(features.select_dtypes(include="number").columns)
        categorical = [c for c in features.columns if c not in numeric]
        return numeric, categorical


    def build_preprocessor(self, frame: pd.DataFrame) -> ColumnTransformer:
        """A ColumnTransformer: median + scaling for numbers, most frequent + one hot for text."""
        numeric, categorical = self.split_columns(frame)
        numeric_pipeline = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)]
        )
