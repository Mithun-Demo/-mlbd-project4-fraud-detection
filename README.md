# Classification Models to Determine Whether a Transaction Is Fraudulent

Project 4, *Machine Learning with Business Data*. Binary classification on two datasets, compared across five
frameworks: **XGBoost, Keras, PyTorch + fastai, PyTorch + TabNet, PyTorch Lightning Flash**.

## Datasets

| Dataset | Source | Rows | Positive class |
| --- | --- | --- | --- |
| Credit card fraud | [`creditcard_sample.csv`](https://github.com/tiwvis/MLBD-Projects/blob/main/data_sets/creditcard_sample.csv) (course repo) | 99,999 (99,618 after dropping 381 duplicates) | `Class = 1` fraud, 223 rows (0.22%) |
| German credit risk | [UCI Statlog German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1,000 | bad credit (coded 2 in the source, recoded to 1), 300 rows |

Put the files in `data/`:

```bash
mkdir -p data
curl -L -o data/creditcard_sample.csv https://raw.githubusercontent.com/tiwvis/MLBD-Projects/main/data_sets/creditcard_sample.csv
# German credit: data/german.csv is included (21 columns, no header, last column = class 1/2)
```

## Run

```bash
pip install -r requirements.txt
python eda.py                       # figures/ + results/eda_summary.txt
for fw in xgboost keras fastai tabnet flash; do
  for ds in credit german; do python run_$fw.py $ds; done
done
python summarize.py                 # results/summary.csv + comparison charts
```

Every `run_*.py` uses `common.py`, so all frameworks get the same cleaned data, the same stratified 80/20 split
(seed 42), the same scaling (fitted on train only) and the same metrics.

## Method in brief

- **Imbalance:** positive-class weight capped at 20 (XGBoost `scale_pos_weight`, weighted loss in Keras, fastai and
  Flash). TabNet uses its built-in balanced sampler instead.
- **Metrics:** precision, recall, F1, ROC-AUC, PR-AUC, accuracy, plus RMSE, MAE and R² computed between the true
  label and the predicted probability. Threshold fixed at 0.5.
- **Note on "regression":** the brief mentions regression and RMSE/R²/MAE, but both targets are binary, so this is
  a classification study; the regression-style metrics are computed on predicted probabilities.

## Results (test set)

F1 / ROC-AUC. Full tables with all metrics are in `results/summary.csv`.

| Framework | Credit card fraud | German credit risk |
| --- | --- | --- |
| XGBoost | 0.902 / 0.974 | 0.607 / 0.787 |
| Keras | 0.765 / 0.967 | 0.581 / 0.765 |
| fastai | 0.771 / 0.979 | 0.589 / 0.753 |
| TabNet | 0.475 / 0.973 | 0.579 / 0.734 |
| Lightning Flash | 0.576 / 0.920 | 0.565 / 0.752 |

The credit card test set has only 45 frauds and the German one 60 bad cases; each number is a single split, so
small gaps are noise.

## Lightning Flash

Lightning Flash 0.8.2 is unmaintained and does not install normally on Python 3.13. It was made to work with
older `pytorch-lightning`, `torchmetrics` and `pytorch-tabular`, plus a small compatibility patch in
`flash_compat.py` (imported first by `run_flash.py`). Install steps are at the bottom of `requirements.txt`.
If you cannot reproduce this, run `run_flash.py` in a fresh Colab with Python 3.10 and `pip install lightning-flash[tabular]`.

## Files

| File | Purpose |
| --- | --- |
| `common.py` | data loading, preprocessing, split, metrics |
| `eda.py` | exploratory data analysis and figures |
| `run_xgboost.py`, `run_keras.py`, `run_fastai.py`, `run_tabnet.py`, `run_flash.py` | one model per framework |
| `flash_compat.py` | compatibility patch for Lightning Flash |
| `summarize.py` | results table and comparison charts |
| `results/` | per-run JSON, `summary.csv`, `eda_summary.txt` |
| `figures/` | EDA and comparison charts |
| `VIVA_PREP.md` | likely viva questions with short answers |
| `Project4_Fraud_Classification.pptx` | presentation |
