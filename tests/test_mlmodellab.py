import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mlmodellab import drift as drift_mod
from mlmodellab import export_mlflow, registry, store, train
from mlmodellab.cli import gate_passes


class LabTests(unittest.TestCase):
    def test_train_one_logreg(self):
        data = train.load_dataset("breast_cancer")
        result = train.train_one("logreg", "c1", data)
        self.assertGreater(result["metrics"]["f1"], 0.9)
        self.assertEqual(result["n_train"], 455)
        self.assertEqual(result["n_test"], 114)

    def test_digits_multiclass(self):
        data = train.load_dataset("digits")
        result = train.train_one("logreg", "c1", data)
        self.assertGreater(result["metrics"]["f1"], 0.85)
        self.assertIsNotNone(result["metrics"]["roc_auc"])

    def test_store_roundtrip_and_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(os.path.join(tmp, "lab.db"))
            data = train.load_dataset("breast_cancer")
            rid = store.record_run(conn, data["name"], data["version"],
                                   data["hash"], "logreg", "c1",
                                   {"C": 1.0},
                                   {"accuracy": 0.97, "f1": 0.97,
                                    "roc_auc": 0.99}, 0.5, 30)
            store.save_reference(conn, rid, np.zeros(30),
                                 np.ones(30))
            self.assertTrue(store.promote(conn, rid, "best F1"))
            champ = store.champion(conn)
            self.assertEqual(champ["id"], rid)
            self.assertEqual(champ["champion_reason"], "best F1")
            self.assertIsNotNone(store.get_reference(conn, rid))

    def test_dataset_versioning(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(os.path.join(tmp, "lab.db"))
            v1, _ = store.register_dataset(conn, "custom", "f.csv",
                                           10, 3, "aaa", ["a", "b"])
            v2, _ = store.register_dataset(conn, "custom", "f2.csv",
                                           12, 3, "bbb", ["a", "b"])
            self.assertEqual((v1, v2), ("v1", "v2"))
            latest = store.get_dataset(conn, "custom")
            self.assertEqual(latest["version"], "v2")
            self.assertEqual(store.get_dataset(conn, "custom",
                                               "v1")["dataset_hash"],
                             "aaa")

    def test_search_small(self):
        data = train.load_dataset("breast_cancer")
        result = train.search_one("logreg", data, n_iter=2)
        self.assertEqual(result["variant"], "search")
        self.assertIn("C", result["params"])
        self.assertEqual(len(result["trials"]), 2)
        self.assertGreater(result["metrics"]["f1"], 0.9)

    def test_drift_clean_vs_shifted(self):
        ref = {"means": [0.0, 0.0], "stds": [1.0, 1.0]}
        clean = drift_mod.check_drift(ref, [[0.1, -0.1]] * 20)
        self.assertEqual(clean["drifted_features"], [])
        shifted = drift_mod.check_drift(ref, [[5.0, 0.0]] * 20)
        self.assertIn(0, shifted["drifted_features"])
        self.assertGreater(shifted["drift_score"],
                           clean["drift_score"])

    def test_gate_logic(self):
        self.assertTrue(gate_passes(0.99, 0.98))
        self.assertFalse(gate_passes(0.98, 0.98))
        self.assertFalse(gate_passes(0.97, 0.98))
        self.assertTrue(gate_passes(0.50, None))

    def test_mlflow_export_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(os.path.join(tmp, "lab.db"))
            data = train.load_dataset("breast_cancer")
            rid = store.record_run(conn, data["name"], data["version"],
                                   data["hash"], "logreg", "c1",
                                   {"C": 1.0},
                                   {"accuracy": 0.97, "f1": 0.97,
                                    "roc_auc": 0.99}, 0.5, 30)
            uuids = export_mlflow.export_runs(conn, tmp,
                                              os.path.join(tmp, "ml"))
            self.assertEqual(len(uuids), 1)
            rdir = os.path.join(tmp, "ml", "0", uuids[0])
            self.assertTrue(os.path.exists(
                os.path.join(rdir, "meta.yaml")))
            self.assertTrue(os.path.exists(
                os.path.join(rdir, "params", "C")))
            with open(os.path.join(rdir, "metrics", "f1")) as fh:
                self.assertIn("0.97", fh.read())

    def test_registry_predict_shape(self):
        data = train.load_dataset("breast_cancer")
        result = train.train_one("logreg", "c1", data)
        with tempfile.TemporaryDirectory() as tmp:
            path = registry.artifact_path(tmp, 1)
            registry.save_model(result["pipeline"], path)
            pipe = registry.load_model(path)
            out = registry.predict_rows(pipe, data["X"][:3])
            self.assertEqual(len(out), 3)
            self.assertIn("prediction", out[0])
            self.assertIn("probability", out[0])

    def test_unknown_model_rejected(self):
        data = train.load_dataset("breast_cancer")
        with self.assertRaises(ValueError):
            train.train_one("nope", "x", data)


if __name__ == "__main__":
    unittest.main()
