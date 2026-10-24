"""One SQLite file: experiment runs, dataset versions, references."""
import hashlib
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    dataset TEXT NOT NULL,
    dataset_version TEXT DEFAULT 'v1',
    dataset_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    variant TEXT NOT NULL,
    params TEXT NOT NULL,
    accuracy REAL,
    f1 REAL,
    roc_auc REAL,
    train_seconds REAL,
    n_features INTEGER,
    champion INTEGER DEFAULT 0,
    champion_reason TEXT
);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    source TEXT NOT NULL,
    target TEXT,
    n_rows INTEGER,
    n_features INTEGER,
    dataset_hash TEXT NOT NULL,
    feature_names TEXT,
    notes TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run_reference (
    run_id INTEGER PRIMARY KEY,
    means TEXT NOT NULL,
    stds TEXT NOT NULL
);
"""


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(runs)")]
    for col, ddl in (("dataset_version",
                      "ALTER TABLE runs ADD COLUMN dataset_version"
                      " TEXT DEFAULT 'v1'"),
                     ("n_features",
                      "ALTER TABLE runs ADD COLUMN n_features INTEGER")):
        if col not in cols:
            conn.execute(ddl)
    conn.commit()
    return conn


def dataset_hash(X, y):
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()[:16]


def register_dataset(conn, name, source, n_rows, n_features, dhash,
                     feature_names, target=None, notes=""):
    count = conn.execute("SELECT COUNT(*) FROM datasets WHERE name = ?",
                         (name,)).fetchone()[0]
    version = "v%d" % (count + 1)
    cur = conn.execute(
        "INSERT INTO datasets (name, version, source, target, n_rows,"
        " n_features, dataset_hash, feature_names, notes, created_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (name, version, source, target, n_rows, n_features, dhash,
         json.dumps(list(feature_names)), notes, time.time()))
    conn.commit()
    return version, cur.lastrowid


def get_dataset(conn, name, version=None):
    if version:
        row = conn.execute("SELECT * FROM datasets WHERE name = ?"
                           " AND version = ?", (name, version)).fetchone()
    else:
        row = conn.execute("SELECT * FROM datasets WHERE name = ?"
                           " ORDER BY id DESC LIMIT 1", (name,)).fetchone()
    return dict(row) if row else None


def list_datasets(conn):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM datasets ORDER BY name, id")]


def record_run(conn, dataset, dataset_version, dhash, model, variant,
               params, metrics, train_seconds, n_features):
    cur = conn.execute(
        "INSERT INTO runs (created_at, dataset, dataset_version,"
        " dataset_hash, model, variant, params, accuracy, f1, roc_auc,"
        " train_seconds, n_features)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), dataset, dataset_version, dhash, model, variant,
         json.dumps(params, sort_keys=True, default=str),
         metrics.get("accuracy"), metrics.get("f1"),
         metrics.get("roc_auc"), train_seconds, n_features))
    conn.commit()
    return cur.lastrowid


def save_reference(conn, run_id, means, stds):
    conn.execute("INSERT OR REPLACE INTO run_reference"
                 " (run_id, means, stds) VALUES (?, ?, ?)",
                 (run_id, json.dumps([float(m) for m in means]),
                  json.dumps([float(s) for s in stds])))
    conn.commit()


def get_reference(conn, run_id):
    row = conn.execute("SELECT * FROM run_reference WHERE run_id = ?",
                       (run_id,)).fetchone()
    if row is None:
        return None
    return {"means": json.loads(row["means"]),
            "stds": json.loads(row["stds"])}


def list_runs(conn, order_by="f1"):
    col = order_by if order_by in ("accuracy", "f1", "roc_auc",
                                   "train_seconds", "created_at") else "f1"
    return [dict(r) for r in conn.execute(
        "SELECT * FROM runs ORDER BY %s DESC" % col)]


def get_run(conn, run_id):
    row = conn.execute("SELECT * FROM runs WHERE id = ?",
                       (run_id,)).fetchone()
    return dict(row) if row else None


def promote(conn, run_id, reason):
    if get_run(conn, run_id) is None:
        return False
    conn.execute("UPDATE runs SET champion = 0")
    conn.execute("UPDATE runs SET champion = 1, champion_reason = ?"
                 " WHERE id = ?", (reason, run_id))
    conn.commit()
    return True


def champion_for_dataset(conn, dataset):
    row = conn.execute("SELECT * FROM runs WHERE champion = 1"
                       " AND dataset = ? ORDER BY id DESC LIMIT 1",
                       (dataset,)).fetchone()
    return dict(row) if row else None


def champion(conn):
    row = conn.execute("SELECT * FROM runs WHERE champion = 1"
                       " ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None
