"""Drift check: incoming rows vs the champion run's training stats."""
import numpy as np


def check_drift(reference, rows, threshold=1.0):
    means = np.array(reference["means"])
    stds = np.array(reference["stds"])
    stds = np.where(stds > 0, stds, 1.0)
    incoming = np.asarray(rows, dtype=float)
    z = np.abs(incoming.mean(axis=0) - means) / stds
    drifted = [int(i) for i, v in enumerate(z) if v > threshold]
    return {"drift_score": round(float(z.mean()), 4),
            "max_z": round(float(z.max()), 4),
            "drifted_features": drifted,
            "n_features": int(len(z)),
            "n_rows": int(incoming.shape[0])}
