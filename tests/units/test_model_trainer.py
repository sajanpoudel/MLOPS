import numpy as np
import pytest

from src.components.model_trainer import candidate_models, train_best_model


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.rand(60, 2)
    y = 3 * x[:, 0] + 2 * x[:, 1] + 1
    return x[:45], y[:45], x[45:], y[45:]


def test_candidate_models_are_named():
    assert set(candidate_models()) == {"linear", "ridge", "forest"}


def test_best_model_is_one_of_the_candidates(data):
    name, model, scores = train_best_model(*data)
    assert name in scores
    assert hasattr(model, "predict")


def test_every_candidate_gets_a_score(data):
    _, _, scores = train_best_model(*data)
    assert set(scores) == {"linear", "ridge", "forest"}


def test_a_linear_target_is_fitted_almost_perfectly(data):
    name, _, scores = train_best_model(*data)
    assert scores[name] > 0.95


def test_the_best_name_has_the_highest_score(data):
    name, _, scores = train_best_model(*data)
    assert scores[name] == max(scores.values())
