"""Training: datasets, metrics, named variants, random search."""
import csv
import time

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from mlmodellab import store

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

SEARCH_SPACES = {
    "logreg": {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
    "rf": {"randomforestclassifier__n_estimators": [100, 200, 300],
           "randomforestclassifier__max_depth": [None, 5, 10, 20],
           "randomforestclassifier__min_samples_split": [2, 5]},
    "gb": {"histgradientboostingclassifier__max_iter": [100, 200, 400],
           "histgradientboostingclassifier__learning_rate":
               [0.03, 0.05, 0.1],
           "histgradientboostingclassifier__max_depth": [None, 3, 5]},
    "svc": {"svc__C": [0.1, 1.0, 10.0],
            "svc__kernel": ["rbf", "linear"]},
}


def load_dataset(name, conn=None, csv_path=None, target=None):
    """Return a dataset dict. Builtins register themselves as v1
    on first sight; CSV datasets are registered by the CLI."""
    if name == "breast_cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        features = list(data.feature_names)
        source = "sklearn:breast_cancer"
    elif name == "digits":
        data = load_digits()
        X, y = data.data, data.target
        features = ["pixel_%d" % i for i in range(X.shape[1])]
        source = "sklearn:digits"
    elif csv_path:
        with open(csv_path, newline="") as fh:
            rows = list(csv.reader(fh))
        header, body = rows[0], rows[1:]
        target_idx = (header.index(target) if target in header
                      else len(header) - 1)
        features = [h for i, h in enumerate(header)
                    if i != target_idx]
        X = np.array([[float(v) for i, v in enumerate(r)
                       if i != target_idx] for r in body])
        y = np.array([float(r[target_idx]) for r in body])
        source = csv_path
    else:
        raise ValueError("unknown dataset: %s (pass a CSV path for"
                         " custom datasets)" % name)
    dhash = store.dataset_hash(X, y)
    version = "v1"
    if conn is not None:
        existing = store.get_dataset(conn, name)
        if existing is None:
            version, _ = store.register_dataset(
                conn, name, source, int(X.shape[0]), int(X.shape[1]),
                dhash, features, target=target)
        else:
            version = existing["version"]
    return {"name": name, "X": X, "y": y, "features": features,
            "hash": dhash, "version": version, "source": source}


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


def _search_estimator(model):
    if model == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000))
    if model == "rf":
        return make_pipeline(RandomForestClassifier(random_state=42,
                                                    n_jobs=-1))
    if model == "gb":
        return make_pipeline(HistGradientBoostingClassifier(
            random_state=42))
    if model == "svc":
        return make_pipeline(StandardScaler(), SVC(random_state=42))
    raise ValueError("unknown model: %s" % model)


def _scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)
    scores = pipe.decision_function(X)
    return np.atleast_2d(scores).T if scores.ndim == 1 else scores


def _metrics(y_test, pred, scores, n_classes):
    acc = round(float(accuracy_score(y_test, pred)), 4)
    if n_classes > 2:
        f1 = round(float(f1_score(y_test, pred, average="macro")), 4)
        try:
            auc = round(float(roc_auc_score(
                y_test, scores, multi_class="ovr", average="macro")), 4)
        except ValueError:
            auc = None
    else:
        f1 = round(float(f1_score(y_test, pred)), 4)
        col = scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
        auc = round(float(roc_auc_score(y_test, col)), 4)
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def _split(data):
    return train_test_split(data["X"], data["y"], test_size=0.2,
                            random_state=42, stratify=data["y"])


def train_one(model, variant, data):
    if model not in MODELS or variant not in MODELS[model]:
        raise ValueError("unknown %s/%s" % (model, variant))
    params = MODELS[model][variant]
    X_train, X_test, y_train, y_test = _split(data)
    pipe = _build(model, params)
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    seconds = round(time.perf_counter() - t0, 3)
    pred = pipe.predict(X_test)
    metrics = _metrics(y_test, pred, _scores(pipe, X_test),
                       len(np.unique(data["y"])))
    return {"model": model, "variant": variant, "params": params,
            "metrics": metrics, "train_seconds": seconds,
            "pipeline": pipe, "feature_names": data["features"],
            "n_train": len(X_train), "n_test": len(X_test),
            "train_means": X_train.mean(axis=0),
            "train_stds": X_train.std(axis=0)}


def search_one(model, data, n_iter=8):
    """Real randomized hyperparameter search, scored by F1."""
    if model not in SEARCH_SPACES:
        raise ValueError("unknown model: %s" % model)
    X_train, X_test, y_train, y_test = _split(data)
    n_classes = len(np.unique(data["y"]))
    scoring = "f1_macro" if n_classes > 2 else "f1"
    search = RandomizedSearchCV(_search_estimator(model),
                                SEARCH_SPACES[model], n_iter=n_iter,
                                scoring=scoring, cv=3, random_state=42,
                                n_jobs=-1)
    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    seconds = round(time.perf_counter() - t0, 3)
    best = search.best_estimator_
    if model == "svc":
        base = best.named_steps["svc"]
        best = make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(base, cv=3).fit(X_train, y_train))
    pred = best.predict(X_test)
    metrics = _metrics(y_test, pred, _scores(best, X_test), n_classes)
    params = {k.split("__", 1)[1]: (v.item() if hasattr(v, "item")
                                    else v)
              for k, v in search.best_params_.items()}
    trials = [{"params": {k.split("__", 1)[1]: v
                          for k, v in row.items()},
               "mean_cv_score": round(float(s), 4)}
              for row, s in zip(search.cv_results_["params"],
                                search.cv_results_["mean_test_score"])]
    return {"model": model, "variant": "search", "params": params,
            "metrics": metrics, "train_seconds": seconds,
            "pipeline": best, "feature_names": data["features"],
            "n_train": len(X_train), "n_test": len(X_test),
            "train_means": X_train.mean(axis=0),
            "train_stds": X_train.std(axis=0), "trials": trials}
