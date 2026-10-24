"""CLI: train, search, list, compare, promote, predict, drift,
gate, export-mlflow, datasets, serve."""
import argparse
import csv
import json
import os
import sys

import numpy as np

from mlmodellab import drift as drift_mod
from mlmodellab import export_mlflow, registry, store, train

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB = os.path.join(BASE, "runs", "lab.db")
DEFAULT_RUNS = os.path.join(BASE, "runs", "artifacts")


def _db_path():
    return os.environ.get("MLMODELLAB_DB", DEFAULT_DB)


def _runs_dir():
    return os.environ.get("MLMODELLAB_RUNS_DIR", DEFAULT_RUNS)


def _conn():
    os.makedirs(os.path.dirname(_db_path()), exist_ok=True)
    return store.connect(_db_path())


def _record(conn, result, data):
    run_id = store.record_run(conn, data["name"], data["version"],
                              data["hash"], result["model"],
                              result["variant"], result["params"],
                              result["metrics"], result["train_seconds"],
                              data["X"].shape[1])
    store.save_reference(conn, run_id, result["train_means"],
                         result["train_stds"])
    registry.save_model(result["pipeline"],
                        registry.artifact_path(_runs_dir(), run_id))
    return run_id


def _resolve_data(args):
    conn = _conn()
    if args.dataset in ("breast_cancer", "digits"):
        return conn, train.load_dataset(args.dataset, conn=conn)
    if not args.csv:
        print("custom dataset needs --csv <file>")
        sys.exit(2)
    data = train.load_dataset(args.dataset, csv_path=args.csv,
                              target=args.target)
    existing = store.get_dataset(conn, args.dataset)
    if existing is None:
        version, _ = store.register_dataset(
            conn, args.dataset, args.csv, int(data["X"].shape[0]),
            int(data["X"].shape[1]), data["hash"], data["features"],
            target=args.target, notes="csv upload")
        data["version"] = version
    else:
        data["version"] = existing["version"]
    return conn, data


def gate_passes(candidate_f1, champion_f1):
    if champion_f1 is None:
        return True
    return candidate_f1 > champion_f1


def cmd_train(args):
    conn, data = _resolve_data(args)
    result = train.train_one(args.model, args.variant, data)
    run_id = _record(conn, result, data)
    print(json.dumps({"run_id": run_id, "dataset": data["name"],
                      "dataset_version": data["version"],
                      "model": args.model, "variant": args.variant,
                      "metrics": result["metrics"],
                      "train_seconds": result["train_seconds"]},
                     indent=2))


def cmd_search(args):
    conn, data = _resolve_data(args)
    result = train.search_one(args.model, data, n_iter=args.trials)
    run_id = _record(conn, result, data)
    print(json.dumps({"run_id": run_id, "dataset": data["name"],
                      "dataset_version": data["version"],
                      "model": args.model, "variant": "search",
                      "best_params": result["params"],
                      "metrics": result["metrics"],
                      "trials": result["trials"]}, indent=2))


def cmd_list(args):
    conn = _conn()
    for r in store.list_runs(conn, args.sort):
        print("%d | %s %s | %s/%s | f1 %.4f acc %.4f auc %s | %ss%s"
              % (r["id"], r["dataset"], r["dataset_version"],
                 r["model"], r["variant"], r["f1"], r["accuracy"],
                 ("%.4f" % r["roc_auc"]) if r["roc_auc"] is not None
                 else "n/a", r["train_seconds"],
                 " | CHAMPION" if r["champion"] else ""))


def cmd_compare(args):
    conn = _conn()
    print(json.dumps(store.list_runs(conn, args.sort), indent=2,
                     default=str))


def cmd_promote(args):
    conn = _conn()
    ok = store.promote(conn, args.run_id, args.reason)
    print("promoted run %d" % args.run_id if ok else "run not found")


def cmd_champion(args):
    conn = _conn()
    champ = store.champion(conn)
    print(json.dumps(champ, indent=2, default=str)
          if champ else "no champion yet")


def _read_rows(path):
    with open(path, newline="") as fh:
        return [[float(v) for v in row] for row in csv.reader(fh)]


def cmd_predict(args):
    conn = _conn()
    run = (store.champion(conn) if args.run_id is None
           else store.get_run(conn, args.run_id))
    if run is None:
        print("no run found; train and promote one first")
        return
    pipe = registry.load_model(registry.artifact_path(_runs_dir(),
                                                      run["id"]))
    out = registry.predict_rows(pipe, np.array(_read_rows(args.rows)))
    print(json.dumps({"run_id": run["id"], "model": run["model"],
                      "predictions": out}, indent=2))


def cmd_drift(args):
    conn = _conn()
    run = (store.champion(conn) if args.run_id is None
           else store.get_run(conn, args.run_id))
    if run is None:
        print("no run found")
        return
    ref = store.get_reference(conn, run["id"])
    if ref is None:
        print("no reference stats for run %d" % run["id"])
        return
    result = drift_mod.check_drift(ref, _read_rows(args.rows))
    result["run_id"] = run["id"]
    print(json.dumps(result, indent=2))


def cmd_gate(args):
    conn = _conn()
    if args.run_id is not None:
        cand = store.get_run(conn, args.run_id)
        if cand is None:
            print("run not found")
            return 2
        cand_f1 = cand["f1"]
        cand_id = cand["id"]
    else:
        _, data = _resolve_data(args)
        result = train.train_one(args.model, args.variant, data)
        cand_id = _record(conn, result, data)
        cand_f1 = result["metrics"]["f1"]
    cand = store.get_run(conn, cand_id)
    champ = store.champion_for_dataset(conn, cand["dataset"])
    if champ is not None and champ["id"] == cand_id:
        print(json.dumps({"verdict": "PASS",
                          "reason": "candidate is the champion",
                          "candidate_run_id": cand_id}, indent=2))
        return 0
    champ_f1 = champ["f1"] if champ else None
    if champ is None:
        print(json.dumps({"verdict": "PASS",
                          "reason": "no champion yet for dataset "
                          "%s" % cand["dataset"],
                          "candidate_run_id": cand_id,
                          "candidate_f1": cand_f1}, indent=2))
        return 0
    ok = gate_passes(cand_f1, champ_f1)
    print(json.dumps({"verdict": "PASS" if ok else "FAIL",
                      "candidate_run_id": cand_id,
                      "candidate_f1": cand_f1,
                      "champion_f1": champ_f1,
                      "reason": ("candidate beats champion" if ok
                                 else "candidate does not beat"
                                 " champion")}, indent=2))
    return 0 if ok else 1


def cmd_dataset_register(args):
    conn = _conn()
    data = train.load_dataset(args.name, csv_path=args.file,
                              target=args.target)
    version, _ = store.register_dataset(
        conn, args.name, args.file, int(data["X"].shape[0]),
        int(data["X"].shape[1]), data["hash"], data["features"],
        target=args.target, notes=args.notes)
    print(json.dumps({"name": args.name, "version": version,
                      "n_rows": int(data["X"].shape[0]),
                      "n_features": int(data["X"].shape[1]),
                      "hash": data["hash"]}, indent=2))


def cmd_datasets(args):
    conn = _conn()
    print(json.dumps(store.list_datasets(conn), indent=2, default=str))


def cmd_export_mlflow(args):
    conn = _conn()
    uuids = export_mlflow.export_runs(conn, _runs_dir(), args.out)
    print(json.dumps({"exported_runs": len(uuids), "out": args.out,
                      "open_with": "mlflow ui --backend-store-uri"
                      " file://%s" % os.path.abspath(args.out)},
                     indent=2))


def cmd_serve(args):
    import uvicorn
    uvicorn.run("mlmodellab.app_api:app", host="127.0.0.1",
                port=args.port)


def main():
    parser = argparse.ArgumentParser(prog="mlmodellab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _dataset_args(p):
        p.add_argument("--dataset", default="breast_cancer")
        p.add_argument("--csv")
        p.add_argument("--target")

    p = sub.add_parser("train")
    p.add_argument("--model", required=True,
                   choices=list(train.MODELS))
    p.add_argument("--variant", required=True)
    _dataset_args(p)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("search")
    p.add_argument("--model", required=True,
                   choices=list(train.MODELS))
    p.add_argument("--trials", type=int, default=8)
    _dataset_args(p)
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("list")
    p.add_argument("--sort", default="f1")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("compare")
    p.add_argument("--sort", default="f1")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("promote")
    p.add_argument("run_id", type=int)
    p.add_argument("--reason", default="")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("champion")
    p.set_defaults(fn=cmd_champion)

    p = sub.add_parser("predict")
    p.add_argument("--rows", required=True)
    p.add_argument("--run-id", type=int, default=None)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("drift")
    p.add_argument("--rows", required=True)
    p.add_argument("--run-id", type=int, default=None)
    p.set_defaults(fn=cmd_drift)

    p = sub.add_parser("gate")
    p.add_argument("--model", choices=list(train.MODELS))
    p.add_argument("--variant")
    p.add_argument("--run-id", type=int, default=None)
    _dataset_args(p)
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("dataset-register")
    p.add_argument("--name", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--target")
    p.add_argument("--notes", default="")
    p.set_defaults(fn=cmd_dataset_register)

    p = sub.add_parser("datasets")
    p.set_defaults(fn=cmd_datasets)

    p = sub.add_parser("export-mlflow")
    p.add_argument("--out", default="mlruns")
    p.set_defaults(fn=cmd_export_mlflow)

    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_serve)

    args = parser.parse_args()
    result = args.fn(args)
    if isinstance(result, int):
        sys.exit(result)


if __name__ == "__main__":
    main()
