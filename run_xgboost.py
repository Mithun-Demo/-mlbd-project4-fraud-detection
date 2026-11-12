"""XGBoost classifier. Usage: python run_xgboost.py credit|german"""
import sys

from xgboost import XGBClassifier

from common import Timer, evaluate, get_split, pos_weight, save_result

ds = sys.argv[1]
X_tr, X_te, y_tr, y_te, _ = get_split(ds)
model = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=pos_weight(y_tr) if ds == "german" else 20,
                      eval_metric="logloss", n_jobs=2, random_state=42)
with Timer() as t:
    model.fit(X_tr, y_tr)
save_result("XGBoost", ds, evaluate(y_te, model.predict_proba(X_te)[:, 1]), t.seconds, loc=9,
            notes="gradient-boosted trees; scale_pos_weight for imbalance")
