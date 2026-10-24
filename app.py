"""ML Model Lab dashboard: leaderboard, run detail, champion."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

from mlmodellab import store

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "runs", "lab.db")


def _conn():
    return store.connect(DB) if os.path.exists(DB) else None


def leaderboard(sort_by):
    conn = _conn()
    if conn is None:
        return [], "No runs yet. Train from the CLI first."
    runs = store.list_runs(conn, sort_by)
    rows = [[r["id"], "%s %s" % (r["dataset"],
             r["dataset_version"] or ""), r["model"], r["variant"],
             "%.4f" % r["accuracy"], "%.4f" % r["f1"],
             ("%.4f" % r["roc_auc"]) if r["roc_auc"] is not None
             else "n/a", "%.2f" % r["train_seconds"],
             "CHAMPION" if r["champion"] else ""]
            for r in runs]
    champ = store.champion(conn)
    note = ("Champion: run %d (%s/%s), F1 %.4f. Reason: %s"
            % (champ["id"], champ["model"], champ["variant"],
               champ["f1"], champ["champion_reason"] or "not recorded")
            if champ else "No champion promoted yet.")
    return rows, note


def detail(run_id):
    conn = _conn()
    if conn is None:
        return "No runs yet."
    run = store.get_run(conn, int(run_id))
    if run is None:
        return "Run not found."
    return ("Run %d: %s/%s\nDataset: %s %s (hash %s)\nParams: %s\n"
            "Accuracy %.4f, F1 %.4f, ROC AUC %.4f, train %.2f s\n"
            "Champion: %s"
            % (run["id"], run["model"], run["variant"], run["dataset"],
               run["dataset_version"] or "",
               run["dataset_hash"], run["params"], run["accuracy"],
               run["f1"], run["roc_auc"], run["train_seconds"],
               "yes, " + (run["champion_reason"] or "")
               if run["champion"] else "no"))


with gr.Blocks(title="ML Model Lab",
               theme=gr.themes.Soft(primary_hue="indigo",
                                    neutral_hue="slate"),
               css=".footer {display: none !important}") as demo:
    gr.Markdown("# ML Model Lab: which run won, and can you prove it?\n\n"
                "Every run is logged with its model, parameters, "
                "metrics, dataset hash, and training time. Promote "
                "one run to champion and predictions come from that "
                "exact saved model file.")
    sort = gr.Dropdown(["f1", "accuracy", "roc_auc"],
                       value="f1", label="Sort leaderboard by")
    table = gr.Dataframe(
        headers=["Run", "Dataset", "Model", "Variant", "Accuracy",
                 "F1", "ROC AUC", "Train (s)", ""],
        datatype=["number", "str", "str", "str", "str", "str", "str",
                  "str", "str"], label="Leaderboard", interactive=False)
    note = gr.Markdown("No runs yet.")
    refresh = gr.Button("Refresh leaderboard")
    refresh.click(leaderboard, inputs=sort, outputs=[table, note])
    run_id = gr.Number(label="Run id for detail", value=1, precision=0)
    detail_btn = gr.Button("Show run detail")
    detail_out = gr.Markdown()
    detail_btn.click(detail, inputs=run_id, outputs=detail_out)
    demo.load(leaderboard, inputs=sort, outputs=[table, note],
              api_name="leaderboard")

if __name__ == "__main__":
    demo.queue()
    demo.launch()
