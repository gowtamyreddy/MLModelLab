# ML Model Lab: plan

## One line
A local-first experiment lab for classic ML: train, compare, and
promote models with full reproducibility, no cloud account, no
paid tracking server.

## Goal
Prove classic ML engineering discipline: experiment tracking,
model comparison, a champion registry, and reproducible inference,
the skills target job descriptions keep asking for under
MLflow, model registry, and MLOps.

## The gap it fills
Her existing ML work (Text Generation, graduate research) shows
models. It does not show the engineering around them: which
experiment won, on what data version, with what parameters, and
whether the winning model can be reloaded and served a month
later. ML Model Lab makes that the product.

## V1 scope
- Train: scikit-learn pipelines over the breast cancer dataset
  (30 features, 569 rows), four model families, named parameter
  variants, fixed seed, stratified split.
- Store: one SQLite file, one row per run: model, parameters,
  metrics (accuracy, F1, ROC AUC), training seconds, dataset
  hash, champion flag.
- Compare: leaderboard sorted by the chosen metric, with
  per-run detail.
- Registry: promote one run to champion, with a reason recorded;
  only one champion at a time.
- Predict: load the champion (or any run) and predict on new
  rows from a CSV, with probabilities.
- Dashboard: Gradio UI with the leaderboard, run detail, and
  the champion card.

## Not in V1
- No MLflow server (the point is a lighter local alternative),
  no cloud deployment, no drift monitoring, no deep learning.

## Role split
- Boone: engine, CLI, dashboard, tests, demo.
- Meganaa: the experiment questions she demos with, which
  metric she would optimize in an interview, README story,
  interview explanation.

## V2 (built 2026-10-24, at her direction: V1 was too simple
to stand out)
- Datasets: breast_cancer + digits (multiclass), plus CSV
  datasets registered with auto-incrementing versions
  (v1, v2, ...) and hashes, not just a hash alone.
- Search: RandomizedSearchCV per model family, F1 scored,
  best trial recorded as a run, full trial table printed.
- Serving: FastAPI app (health, runs, datasets, champion,
  predict, drift) for the champion model.
- Drift: per-feature standardized mean shift against the
  champion run's stored training means and stds.
- CI gate: a candidate must strictly beat the champion on
  F1 or the command exits 1. Ties fail. This is the
  TrustLens idea applied to classic ML.
- MLflow export: standard file-store layout MLflow UI can
  open, produced without installing MLflow.
