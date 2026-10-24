# ML Model Lab

Train classic ML models, compare them honestly, and promote one
to champion, with every run recorded and reproducible.

ML Model Lab is a local-first experiment lab. It trains
scikit-learn pipelines on the breast cancer dataset, logs each
run (model, parameters, metrics, dataset hash, training time)
into one SQLite file, ranks runs on a leaderboard, lets you
promote a champion, and serves predictions from the champion
model file. No cloud account, no paid tracking server.

## Run it

    python -m mlmodellab.cli train --model logreg --variant c1
    python -m mlmodellab.cli train --model rf --variant deep
    python -m mlmodellab.cli compare
    python -m mlmodellab.cli promote <run_id> --reason "best F1"
    python -m mlmodellab.cli predict --rows data/sample_rows.csv

Or open the dashboard:

    python app.py

## Status
V1: train, compare, promote, predict, dashboard, tests.
