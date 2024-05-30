"""Trains a few regression models and keeps the best one."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from src.components.model_evaluation import r2_score
