# ML Model Lab

Train classic ML models, compare them honestly, and promote one
to champion, with every run recorded and reproducible.

## V1: the lab

    python -m mlmodellab.cli train --model logreg --variant c1
    python -m mlmodellab.cli train --model rf --variant deep
    python -m mlmodellab.cli compare
    python -m mlmodellab.cli promote <run_id> --reason "best F1"
    python -m mlmodellab.cli predict --rows data/sample_rows.csv
    python app.py   # leaderboard dashboard

## V2: the expansion

More datasets with real versioning, randomized hyperparameter
search, a FastAPI serving endpoint, drift checks, a CI gate, and
an MLflow-compatible export.

    # a second dataset, multiclass
    python -m mlmodellab.cli train --model logreg --variant c1 \
        --dataset digits

    # register your own CSV as a versioned dataset
    # (last column is the target unless --target names one)
    python -m mlmodellab.cli dataset-register --name churn \
        --file data/churn.csv
    python -m mlmodellab.cli datasets

    # real search instead of hand-picked variants
    python -m mlmodellab.cli search --model rf --trials 12

    # serve the champion over HTTP
    python -m mlmodellab.cli serve --port 8000
    # GET /health, GET /champion, GET /runs
    # POST /predict {"rows": [[...30 numbers...]]}
    # POST /drift {"rows": [[...]]}

    # drift: incoming rows vs the champion's training stats
    python -m mlmodellab.cli drift --rows data/new_rows.csv

    # CI gate: candidate must BEAT the champion on F1, ties fail
    python -m mlmodellab.cli gate --model gb --variant slow
    # exit 0 on PASS, exit 1 on FAIL

    # MLflow-compatible export (open it with a local MLflow UI)
    python -m mlmodellab.cli export-mlflow --out mlruns
    # mlflow ui --backend-store-uri file://$PWD/mlruns

## CI gate workflow (add via the GitHub web UI)

```yaml
name: champion-gate
on: [push]
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install scikit-learn joblib
      - run: python -m mlmodellab.cli gate --model logreg --variant c1
```

## Status
V1 done and verified. V2 done and verified: datasets, search,
serving, drift, gate, MLflow export, tests.
