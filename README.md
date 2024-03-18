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
