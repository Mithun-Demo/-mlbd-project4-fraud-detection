"""Exploratory data analysis for both datasets. Saves figures to ./figures and a text summary to ./results/eda_summary.txt."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from common import FIGS, RESULTS, load_raw

sns.set_theme(style="whitegrid")
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ---------------------------------------------------------------- credit card
df = load_raw("credit")
log("=== CREDIT CARD DATASET ===")
log(f"shape: {df.shape}; missing values: {int(df.isna().sum().sum())}; duplicate rows: {int(df.duplicated().sum())}")
vc = df["Class"].value_counts()
log(f"class counts: legit={vc[0]}, fraud={vc[1]} ({vc[1] / len(df) * 100:.3f}% fraud)")
log("Amount by class:\n" + df.groupby("Class")["Amount"].describe().round(2).to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(x="Class", data=df, ax=ax[0])
ax[0].set_yscale("log")
ax[0].set_title("Class balance (log scale)")
for c, col in zip([0, 1], ["tab:blue", "tab:red"]):
    ax[1].hist(np.log1p(df.loc[df.Class == c, "Amount"]), bins=40, alpha=.6, density=True, color=col,
               label="fraud" if c else "legit")
ax[1].set_title("log(1 + Amount) by class")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{FIGS}/credit_class_amount.png", dpi=150)
plt.close()

corr = df.corr()["Class"].drop("Class").sort_values()
log("Top feature correlations with Class:\n" + corr.reindex(corr.abs().sort_values(ascending=False).index)[:8].round(3).to_string())
plt.figure(figsize=(8, 6))
corr.plot(kind="barh", color=["tab:red" if v < 0 else "tab:blue" for v in corr])
plt.title("Correlation of each feature with Class (fraud)")
plt.tight_layout()
plt.savefig(f"{FIGS}/credit_corr_with_class.png", dpi=150)
plt.close()

top = corr.abs().sort_values(ascending=False).index[:6]
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for a, c in zip(axes.ravel(), top):
    sns.boxplot(x="Class", y=c, data=df, ax=a, showfliers=False)
    a.set_title(c)
plt.tight_layout()
plt.savefig(f"{FIGS}/credit_top_features_boxplot.png", dpi=150)
plt.close()

hour = (df["Time"] // 3600) % 24
rate = df.groupby(hour)["Class"].mean() * 100
plt.figure(figsize=(7, 3.5))
rate.plot(marker="o")
plt.ylabel("fraud rate (%)")
plt.xlabel("hour of day (from Time)")
plt.title("Fraud rate by hour")
plt.tight_layout()
plt.savefig(f"{FIGS}/credit_fraud_by_hour.png", dpi=150)
plt.close()

# ---------------------------------------------------------------- german credit
g = load_raw("german")
log("\n=== GERMAN CREDIT DATASET ===")
log(f"shape: {g.shape}; missing values: {int(g.isna().sum().sum())}")
vc = g["Class"].value_counts()
log(f"class counts: good={vc[0]}, bad={vc[1]} ({vc[1] / len(g) * 100:.1f}% bad)")
log("numeric summary:\n" + g[["duration", "amount", "age"]].describe().round(1).to_string())

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, c in zip(ax, ["duration", "amount", "age"]):
    sns.boxplot(x="Class", y=c, data=g, ax=a)
    a.set_title(f"{c} by class (1 = bad credit)")
plt.tight_layout()
plt.savefig(f"{FIGS}/german_numeric_by_class.png", dpi=150)
plt.close()

for c in ["status", "credit_history", "savings"]:
    r = g.groupby(c)["Class"].mean().sort_values(ascending=False) * 100
    log(f"bad-credit rate by {c} (%):\n" + r.round(1).to_string())
r = g.groupby("status")["Class"].mean().sort_values() * 100
plt.figure(figsize=(6, 3.5))
r.plot(kind="barh", color="tab:orange")
plt.xlabel("bad-credit rate (%)")
plt.title("Bad-credit rate by checking-account status code")
plt.tight_layout()
plt.savefig(f"{FIGS}/german_status_rate.png", dpi=150)
plt.close()

with open(f"{RESULTS}/eda_summary.txt", "w") as f:
    f.write("\n".join(lines))
