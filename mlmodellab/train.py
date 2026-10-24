"""Training: fixed split, four model families, named variants."""
import time

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

MODELS = {
    "logreg": {
        "c1": {"C": 1.0},
        "c01": {"C": 0.1},
    },
    "rf": {
        "shallow": {"n_estimators": 100, "max_depth": 5},
        "deep": {"n_estimators": 200, "max_depth": None},
    },
    "gb": {
        "default": {"max_iter": 200, "learning_rate": 0.1},
        "slow": {"max_iter": 400, "learning_rate": 0.05},
    },
    "svc": {
        "rbf": {"C": 1.0, "kernel": "rbf"},
    },
}


def _build(model, params):
    if model == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=params["C"],
                                                max_iter=2000))
    if model == "rf":
        return make_pipeline(RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"], random_state=42, n_jobs=-1))
    if model == "gb":
        return make_pipeline(HistGradientBoostingClassifier(
            max_iter=params["max_iter"],
            learning_rate=params["learning_rate"], random_state=42))
    if model == "svc":
        base = SVC(C=params["C"], kernel=params["kernel"],
                   random_state=42)
        return make_pipeline(StandardScaler(),
                             CalibratedClassifierCV(base, cv=3))
    raise ValueError("unknown model: %s" % model)


def load_data():
    data = load_breast_cancer()
    return data.data, data.target, data.feature_names


def train_one(model, variant, db_dir=None):
    if model not in MODELS or variant not in MODELS[model]:
        raise ValueError("unknown %s/%s" % (model, variant))
    params = MODELS[model][variant]
    X, y, features = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = _build(model, params)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    seconds = round(time.perf_counter() - t0, 3)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    return {"model": model, "variant": variant, "params": params,
            "metrics": metrics, "train_seconds": seconds,
            "pipeline": pipe, "feature_names": list(features),
            "n_train": len(X_train), "n_test": len(X_test)}
