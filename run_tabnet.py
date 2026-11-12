"""PyTorch + TabNet (pytorch-tabnet). Usage: python run_tabnet.py credit|german"""
import sys

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split

from common import Timer, evaluate, get_split, save_result

ds = sys.argv[1]
torch.manual_seed(42)
np.random.seed(42)
X_tr, X_te, y_tr, y_te, cols = get_split(ds)
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.1, stratify=y_tr, random_state=42)

# German Credit has only ~720 training rows: use small batches and more epochs, otherwise one epoch is a
# single gradient step and early stopping fires long before the model has learned anything.
small = ds == "german"
model = TabNetClassifier(n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=42, device_name="cpu", verbose=0)
with Timer() as t:
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], eval_metric=["auc"],
              max_epochs=100 if small else 30, patience=15 if small else 6,
              batch_size=64 if small else 1024, virtual_batch_size=16 if small else 128,
              weights=1)  # weights=1 -> balanced sampling of the minority class
proba = model.predict_proba(X_te)[:, 1]
res = evaluate(y_te, proba)
save_result("TabNet", ds, res, t.seconds, loc=13,
            notes="TabNet (n_d=n_a=8, 3 steps), balanced sampler, early stopping on val AUC")

# TabNet's built-in feature importance (the 'transparency' angle); useful for the German data, whose
# columns have real meaning
imp = sorted(zip(cols, model.feature_importances_), key=lambda x: -x[1])[:8]
print("top TabNet features:", [(c, round(float(v), 3)) for c, v in imp])
