"""Builds the preprocessing that turns raw columns into model features."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    """Which column is predicted."""

    target_column: str = "target"


def none_to_nan(values):
    """Turn None into NaN so text columns are imputed the same way on every pandas version."""
    frame = pd.DataFrame(values).astype(object)
    return frame.where(frame.notna(), np.nan)


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
                ("missing", FunctionTransformer(none_to_nan)),
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            [("numeric", numeric_pipeline, numeric), ("categorical", categorical_pipeline, categorical)]
        )


    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame):
        """Fit the preprocessor on train and return (X_train, y_train, X_test, y_test, preprocessor)."""
        target = self.config.target_column
        preprocessor = self.build_preprocessor(train)
        x_train = preprocessor.fit_transform(train.drop(columns=[target]))
        x_test = preprocessor.transform(test.drop(columns=[target]))
        return x_train, np.asarray(train[target]), x_test, np.asarray(test[target]), preprocessor
