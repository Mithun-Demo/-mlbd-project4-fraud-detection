"""Collect results/*.json into results/summary.csv and draw the comparison charts."""
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FIGS, RESULTS

rows = [json.load(open(f)) for f in glob.glob(f"{RESULTS}/*_*.json")]
df = pd.DataFrame(rows)
order = ["XGBoost", "Keras", "fastai", "TabNet", "Lightning Flash"]
df["framework"] = pd.Categorical(df["framework"], order, ordered=True)
df = df.sort_values(["dataset", "framework"])
cols = ["framework", "dataset", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "rmse", "mae", "r2",
        "fit_seconds", "lines_of_code"]
df[cols].to_csv(f"{RESULTS}/summary.csv", index=False)
print(df[cols].to_string(index=False))

titles = {"credit": "Credit card fraud (test set)", "german": "German credit risk (test set)"}
for metric, fname in [("f1", "compare_f1"), ("roc_auc", "compare_auc")]:
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, ds in zip(ax, ["credit", "german"]):
        d = df[df.dataset == ds]
        bars = a.bar(d["framework"].astype(str), d[metric], color=["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5", "#c53030"])
        a.bar_label(bars, fmt="%.2f", fontsize=8)
        a.set_title(titles[ds])
        a.set_ylim(0, 1.05)
        a.tick_params(axis="x", rotation=25)
    ax[0].set_ylabel(metric.upper().replace("_", "-"))
    plt.tight_layout()
    plt.savefig(f"{FIGS}/{fname}.png", dpi=150)
    plt.close()
