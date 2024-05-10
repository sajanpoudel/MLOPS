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
