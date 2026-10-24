"""CLI: train, list, compare, promote, predict, champion."""
import argparse
import csv
import json
import os

import numpy as np

from mlmodellab import registry, store, train

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "runs", "lab.db")
RUNS_DIR = os.path.join(BASE, "runs", "artifacts")


def _conn():
    os.makedirs(os.path.join(BASE, "runs"), exist_ok=True)
    return store.connect(DB)


def cmd_train(args):
    result = train.train_one(args.model, args.variant)
    X, y, _ = train.load_data()
    dhash = store.dataset_hash(X, y)
    conn = _conn()
    run_id = store.record_run(conn, "breast_cancer", dhash,
                              args.model, args.variant, result["params"],
                              result["metrics"], result["train_seconds"])
    registry.save_model(result["pipeline"],
                        registry.artifact_path(RUNS_DIR, run_id))
    print(json.dumps({"run_id": run_id, "model": args.model,
                      "variant": args.variant,
                      "metrics": result["metrics"],
                      "train_seconds": result["train_seconds"]}, indent=2))


def cmd_list(args):
    conn = _conn()
    for r in store.list_runs(conn, args.sort):
        print("%d | %s/%s | f1 %.4f acc %.4f auc %.4f | %ss%s"
              % (r["id"], r["model"], r["variant"], r["f1"],
                 r["accuracy"], r["roc_auc"], r["train_seconds"],
                 " | CHAMPION" if r["champion"] else ""))


def cmd_compare(args):
    conn = _conn()
    runs = store.list_runs(conn, args.sort)
    print(json.dumps(runs, indent=2, default=str))


def cmd_promote(args):
    conn = _conn()
    ok = store.promote(conn, args.run_id, args.reason)
    print("promoted run %d" % args.run_id if ok else "run not found")


def cmd_champion(args):
    conn = _conn()
    champ = store.champion(conn)
    print(json.dumps(champ, indent=2, default=str)
          if champ else "no champion yet")


def cmd_predict(args):
    conn = _conn()
    run = (store.champion(conn) if args.run_id is None
           else store.get_run(conn, args.run_id))
    if run is None:
        print("no run found; train and promote one first")
        return
    with open(args.rows, newline="") as fh:
        rows = [[float(v) for v in row] for row in csv.reader(fh)]
    pipe = registry.load_model(registry.artifact_path(RUNS_DIR,
                                                      run["id"]))
    out = registry.predict_rows(pipe, np.array(rows))
    print(json.dumps({"run_id": run["id"], "model": run["model"],
                      "predictions": out}, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="mlmodellab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--model", required=True,
                   choices=list(train.MODELS))
    p.add_argument("--variant", required=True)
    p.set_defaults(fn=cmd_train)

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

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
