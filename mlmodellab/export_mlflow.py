"""MLflow-compatible export: a file store MLflow UI can open.

Writes the standard file-store layout (mlruns/<experiment>/<run>/
with meta.yaml, params/, metrics/, tags/, artifacts/) so a team
already running MLflow can point `mlflow ui --backend-store-uri`
at the export and browse these runs. No mlflow install needed
to produce it.
"""
import hashlib
import json
import os
import shutil
import time


def _run_uuid(run_id):
    return hashlib.md5(("mlmodellab-%d" % run_id).encode()).hexdigest()


def export_runs(conn, runs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    exp_dir = os.path.join(out_dir, "0")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "meta.yaml"), "w") as fh:
        fh.write("name: ml-model-lab\nartifact_location: %s\n"
                 % exp_dir)
    runs = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    exported = []
    for row in runs:
        run = dict(row)
        uuid = _run_uuid(run["id"])
        rdir = os.path.join(exp_dir, uuid)
        for sub in ("params", "metrics", "tags", "artifacts"):
            os.makedirs(os.path.join(rdir, sub), exist_ok=True)
        ts = int(run["created_at"] * 1000)
        with open(os.path.join(rdir, "meta.yaml"), "w") as fh:
            fh.write("run_uuid: %s\n" % uuid)
            fh.write("name: %s/%s\n" % (run["model"], run["variant"]))
            fh.write("source: mlmodellab\n")
            fh.write("entry_point_name: \n")
            fh.write("status: FINISHED\n")
            fh.write("start_time: %d\n" % ts)
            fh.write("end_time: %d\n" % ts)
            fh.write("lifecycle_stage: active\n")
            fh.write("artifact_uri: %s\n"
                     % os.path.join(rdir, "artifacts"))
        for key, value in json.loads(run["params"]).items():
            with open(os.path.join(rdir, "params", str(key)), "w") as fh:
                fh.write(str(value))
        for metric in ("accuracy", "f1", "roc_auc"):
            if run[metric] is not None:
                with open(os.path.join(rdir, "metrics", metric),
                          "w") as fh:
                    fh.write("%d %s\n" % (ts, run[metric]))
        with open(os.path.join(rdir, "tags",
                               "mlflow.source.name"), "w") as fh:
            fh.write("mlmodellab")
        with open(os.path.join(rdir, "tags", "dataset"), "w") as fh:
            fh.write("%s %s" % (run["dataset"],
                                run["dataset_version"] or ""))
        art = os.path.join(runs_dir, "run_%d.joblib" % run["id"])
        if os.path.exists(art):
            shutil.copy(art, os.path.join(rdir, "artifacts",
                                          "model.joblib"))
        exported.append(uuid)
    return exported
