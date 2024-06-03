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


def train_best_model(x_train, y_train, x_test, y_test, models: dict = None):
    """Fit every model on the training data and return (name, model, scores).

    scores maps each model name to its r2 on the test data. The best score wins.
    """
    models = models if models is not None else candidate_models()
    scores = {}
    fitted = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores[name] = r2_score(list(y_test), list(model.predict(x_test)))
        fitted[name] = model
    best = max(scores, key=scores.get)
    return best, fitted[best], scores
