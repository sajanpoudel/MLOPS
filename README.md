# MLOPS

A starter layout for a machine learning project with separate components,
pipelines, a logger and a custom exception.

## Layout

```
src/
  components/   data ingestion, transformation, training and evaluation
  pipeline/     training and prediction pipelines
  logger/       file logger, writes to logs/
  exception/    CustomException with file and line details
  utils/        save_object and load_object helpers
tests/units     unit tests
template.py     creates any missing file from the layout above
```

## Setup

```
./init_setup.sh
source env/bin/activate
pytest
```

## Training a model

```python
from src.pipeline.training_pipeline import run_training
from src.pipeline.prediction_pipeline import predict

metrics = run_training("data.csv", target_column="price", model_path="artifacts/model.pkl")
print(metrics)   # {'mae': ..., 'rmse': ..., 'r2': ..., 'model': 'forest'}

predict("artifacts/model.pkl", [{"size": 120, "city": "a"}])
```

The pipeline reads the csv, splits it 80/20, imputes and scales numbers, one hot encodes text, compares a linear model, a ridge model and a random forest, and saves the best one with its preprocessor.
