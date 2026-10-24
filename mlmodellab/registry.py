"""Artifact storage: one model file per run, champion resolution."""
import os

import joblib


def artifact_path(runs_dir, run_id):
    return os.path.join(runs_dir, "run_%d.joblib" % run_id)


def save_model(pipe, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipe, path)


def load_model(path):
    return joblib.load(path)


def predict_rows(pipe, rows):
    proba = pipe.predict_proba(rows)
    preds = pipe.predict(rows)
    out = []
    for p, pr in zip(preds, proba):
        prob = float(pr[1]) if len(pr) == 2 else float(max(pr))
        out.append({"prediction": int(p), "probability": round(prob, 4)})
    return out
