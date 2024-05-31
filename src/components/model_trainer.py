"""Trains a few regression models and keeps the best one."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.components.model_evaluation import r2_score


def candidate_models(random_state: int = 42) -> dict:
    """The models that are compared."""
    return {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "forest": RandomForestRegressor(n_estimators=50, random_state=random_state),
    }
