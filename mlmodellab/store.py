"""One SQLite file holding every experiment run."""
import hashlib
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    dataset TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    variant TEXT NOT NULL,
    params TEXT NOT NULL,
    accuracy REAL,
    f1 REAL,
    roc_auc REAL,
    train_seconds REAL,
    champion INTEGER DEFAULT 0,
    champion_reason TEXT
)
"""


def connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def dataset_hash(X, y):
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    return h.hexdigest()[:16]


def record_run(conn, dataset, dhash, model, variant, params, metrics,
               train_seconds):
    cur = conn.execute(
        "INSERT INTO runs (created_at, dataset, dataset_hash, model,"
        " variant, params, accuracy, f1, roc_auc, train_seconds)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (time.time(), dataset, dhash, model, variant,
         json.dumps(params, sort_keys=True), metrics["accuracy"],
         metrics["f1"], metrics["roc_auc"], train_seconds))
    conn.commit()
    return cur.lastrowid


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


def champion(conn):
    row = conn.execute("SELECT * FROM runs WHERE champion = 1"
                       " ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row else None
