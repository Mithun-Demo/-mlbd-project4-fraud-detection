"""PyTorch Lightning Flash tabular classifier. Usage: python run_flash.py credit|german"""
import sys

import flash_compat  # noqa: F401  (must come before `import flash`; see flash_compat.py)
import numpy as np
import pandas as pd
import torch
import flash
from flash.tabular import TabularClassificationData, TabularClassifier
from sklearn.model_selection import train_test_split

from functools import partial

import torch.nn.functional as F

from common import Timer, evaluate, get_split, pos_weight, save_result

ds = sys.argv[1]
torch.manual_seed(42)
np.random.seed(42)
X_tr, X_te, y_tr, y_te, cols = get_split(ds)
cols = [f"f{i}" for i in range(len(cols))]
tr = pd.DataFrame(X_tr, columns=cols).assign(Class=y_tr)
te = pd.DataFrame(X_te, columns=cols).assign(Class=y_te)
fit_df, val_df = train_test_split(tr, test_size=0.1, stratify=tr["Class"], random_state=42)

dm = TabularClassificationData.from_data_frame(
    numerical_fields=cols, target_fields="Class", train_data_frame=fit_df, val_data_frame=val_df,
    predict_data_frame=te.drop(columns="Class"), batch_size=64 if ds == "german" else 256, num_workers=0)
w = min(pos_weight(y_tr), 20.0)  # same capped class weight as the other frameworks
model = TabularClassifier.from_data(dm, backbone="tabnet", learning_rate=2e-2,
                                    loss_fn=partial(F.cross_entropy, weight=torch.tensor([1.0, w])))
trainer = flash.Trainer(max_epochs=60 if ds == "german" else 5, accelerator="cpu", enable_progress_bar=False, logger=False,
                        enable_checkpointing=False)
with Timer() as t:
    trainer.fit(model, datamodule=dm)
preds = trainer.predict(model, datamodule=dm, output="probabilities")  # list of per-batch outputs
proba = np.concatenate([np.asarray([p[1] if hasattr(p, "__len__") else p for p in b]) for b in preds])
save_result("Lightning Flash", ds, evaluate(y_te, proba), t.seconds, loc=15,
            notes="Flash TabularClassifier (TabNet backbone), weighted cross-entropy")
