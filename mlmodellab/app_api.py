"""FastAPI serving endpoint for the champion model."""
import os

import numpy as np
from fastapi import FastAPI, HTTPException

from mlmodellab import drift as drift_mod
from mlmodellab import registry, store

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI(title="ML Model Lab API")


def _db_path():
    return os.environ.get("MLMODELLAB_DB",
                          os.path.join(BASE, "runs", "lab.db"))


def _runs_dir():
    return os.environ.get("MLMODELLAB_RUNS_DIR",
                          os.path.join(BASE, "runs", "artifacts"))


def _conn():
    return store.connect(_db_path())


def _champion_pipe():
    conn = _conn()
    champ = store.champion(conn)
    if champ is None:
        return None, None
    path = registry.artifact_path(_runs_dir(), champ["id"])
    if not os.path.exists(path):
        return champ, None
    return champ, registry.load_model(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/runs")
def runs(sort: str = "f1"):
    return store.list_runs(_conn(), sort)


@app.get("/datasets")
def datasets():
    return store.list_datasets(_conn())


@app.get("/champion")
def champion():
    champ = store.champion(_conn())
    if champ is None:
        raise HTTPException(status_code=404, detail="no champion yet")
    return champ


@app.post("/predict")
def predict(body: dict):
    rows = body.get("rows")
    if not rows:
        raise HTTPException(status_code=400, detail="rows required")
    champ, pipe = _champion_pipe()
    if champ is None or pipe is None:
        raise HTTPException(status_code=404,
                            detail="no champion model available")
    try:
        out = registry.predict_rows(pipe, np.array(rows, dtype=float))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"run_id": champ["id"], "model": champ["model"],
            "predictions": out}


@app.post("/drift")
def drift(body: dict):
    rows = body.get("rows")
    if not rows:
        raise HTTPException(status_code=400, detail="rows required")
    conn = _conn()
    run_id = body.get("run_id")
    run = (store.get_run(conn, run_id) if run_id
           else store.champion(conn))
    if run is None:
        raise HTTPException(status_code=404, detail="no run found")
    ref = store.get_reference(conn, run["id"])
    if ref is None:
        raise HTTPException(status_code=404,
                            detail="no reference stats for run")
    result = drift_mod.check_drift(ref, rows)
    result["run_id"] = run["id"]
    return result
