# Project Guide

## Overview
This project implements a robust hybrid intrusion detection system (IDS) with supervised and anomaly models, adversarial testing, and ablation studies.

## Prerequisites
- Python 3.10+
- Dataset files placed in the data folder:
  - data/UNSW_NB15_training-set.csv
  - data/UNSW_NB15_testing-set.csv

## Setup
1. Create and activate a virtual environment.

  macOS:
  - python3 -m venv .venv
  - source .venv/bin/activate

  Windows (PowerShell):
  - py -m venv .venv
  - .venv\Scripts\Activate.ps1

2. Install dependencies:
  - pip install -r requirements.txt

3. Verify or update configuration:
  - Edit config.yaml to change dataset paths, target column, or model settings.

## Run the Pipeline
macOS:
- Debug run (dev mode):
  - python -X dev -m src.main --config config.yaml
- Normal run:
  - python -m src.main --config config.yaml

Windows (PowerShell):
- Debug run (dev mode):
  - python -X dev -m src.main --config config.yaml
- Normal run:
  - python -m src.main --config config.yaml

## Outputs
- artifacts/reports/metrics.json: metrics for clean, adversarial, and ablation runs
- artifacts/models/hybrid_detector.joblib: trained hybrid detector

## Configuration Tips
- Change supervised model via model.supervised (random_forest or gradient_boosting)
- Change anomaly model via model.anomaly (isolation_forest or one_class_svm)
- Toggle calibration via model.calibrate
- Adjust adversarial parameters in adversarial section
- Adjust threshold selection via evaluation.threshold_target_fpr

## Troubleshooting
- If you see errors about missing columns, update dataset.target_col or drop columns in config.yaml.
- If categorical columns are not detected correctly, list them in dataset.categorical_cols.
- On Windows, if activation fails, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
