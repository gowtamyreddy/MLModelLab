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
