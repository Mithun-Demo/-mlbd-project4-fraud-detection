"""PyTorch + fastai tabular learner. Usage: python run_fastai.py credit|german"""
import sys

import numpy as np
import pandas as pd
import torch
from fastai.tabular.all import (CategoryBlock, CrossEntropyLossFlat, RandomSplitter, TabularPandas,
                                accuracy, tabular_learner)

from common import Timer, evaluate, get_split, pos_weight, save_result

ds = sys.argv[1]
torch.manual_seed(42)
np.random.seed(42)
X_tr, X_te, y_tr, y_te, cols = get_split(ds)
cols = [f"f{i}" for i in range(len(cols))]
train_df = pd.DataFrame(X_tr, columns=cols).assign(Class=y_tr)
test_df = pd.DataFrame(X_te, columns=cols)

splits = RandomSplitter(valid_pct=0.1, seed=42)(range(len(train_df)))
to = TabularPandas(train_df, procs=[], cat_names=[], cont_names=cols, y_names="Class",
                   y_block=CategoryBlock(), splits=splits)
dls = to.dataloaders(bs=256)
w = min(pos_weight(y_tr), 20.0)  # same capped class weight as XGBoost / Keras
learn = tabular_learner(dls, layers=[64, 32], metrics=accuracy,
                        loss_func=CrossEntropyLossFlat(weight=torch.tensor([1.0, w])))
with Timer() as t:
    learn.fit_one_cycle(8, 1e-2)
probs, _ = learn.get_preds(dl=dls.test_dl(test_df))
save_result("fastai", ds, evaluate(y_te, probs[:, 1].numpy()), t.seconds, loc=14,
            notes="fastai tabular_learner, layers [64, 32], one-cycle policy, weighted CE loss")
