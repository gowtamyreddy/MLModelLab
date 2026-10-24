import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from mlmodellab import registry, store, train


class LabTests(unittest.TestCase):
    def test_train_one_logreg(self):
        result = train.train_one("logreg", "c1")
        self.assertGreater(result["metrics"]["f1"], 0.9)
        self.assertEqual(result["n_train"], 455)
        self.assertEqual(result["n_test"], 114)

    def test_store_roundtrip_and_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(os.path.join(tmp, "lab.db"))
            X, y, _ = train.load_data()
            dhash = store.dataset_hash(X, y)
            rid = store.record_run(conn, "breast_cancer", dhash,
                                   "logreg", "c1", {"C": 1.0},
                                   {"accuracy": 0.97, "f1": 0.97,
                                    "roc_auc": 0.99}, 0.5)
            self.assertTrue(store.promote(conn, rid, "best F1"))
            champ = store.champion(conn)
            self.assertEqual(champ["id"], rid)
            self.assertEqual(champ["champion_reason"], "best F1")

    def test_promote_unknown_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = store.connect(os.path.join(tmp, "lab.db"))
            self.assertFalse(store.promote(conn, 999, "nope"))

    def test_registry_predict_shape(self):
        result = train.train_one("logreg", "c1")
        with tempfile.TemporaryDirectory() as tmp:
            path = registry.artifact_path(tmp, 1)
            registry.save_model(result["pipeline"], path)
            pipe = registry.load_model(path)
            X, y, _ = train.load_data()
            out = registry.predict_rows(pipe, X[:3])
            self.assertEqual(len(out), 3)
            self.assertIn("prediction", out[0])
            self.assertIn("probability", out[0])

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            train.train_one("nope", "x")


if __name__ == "__main__":
    unittest.main()
