#!/usr/bin/env bash
set -e
python -m src.features.build_features --in data/raw/matches.csv --out data/processed/train.parquet
python -m src.models.train --in data/processed/train.parquet --model data/models/model.joblib
python -m src.models.evaluate --in data/processed/train.parquet --model data/models/model.joblib --holdout_weeks 5
python -m src.scrape.example_provider --out data/raw/upcoming.csv
python -m src.models.predict --model data/models/model.joblib --fixtures data/raw/upcoming.csv --out data/processed/predictions.csv
