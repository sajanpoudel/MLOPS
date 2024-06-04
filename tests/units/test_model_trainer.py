import numpy as np
import pytest

from src.components.model_trainer import candidate_models, train_best_model


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    x = rng.rand(60, 2)
    y = 3 * x[:, 0] + 2 * x[:, 1] + 1
    return x[:45], y[:45], x[45:], y[45:]
