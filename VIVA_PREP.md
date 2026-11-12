# Viva preparation: Project 4, transaction fraud classification

Short answers you can say out loud. Numbers are from `results/summary.csv`.

**1. What problem are you solving?**
Predicting whether a transaction is fraud (credit card data) or whether an applicant is a bad credit risk
(German data). Both are binary classification with `Class` as the target.

**2. The brief says regression and RMSE, R², MAE. Why classification?**
The target is 0/1, so it is a classification problem. To cover the brief's metrics I also computed RMSE, MAE and R²
between the true label and the predicted probability. (RMSE squared is the Brier score.)

**3. Why not use accuracy?**
Only 0.22% of credit card transactions are fraud, so predicting "not fraud" every time scores 99.8%. I used
precision, recall, F1 and ROC-AUC instead.

**4. What do precision and recall mean here?**
Precision: of the transactions we flag, how many are really fraud. Recall: of all real frauds, how many we catch.
XGBoost on credit card data: precision 1.00 (no false alarms), recall 0.82 (37 of 45 frauds).

**5. How did you handle class imbalance?**
Stratified split, plus a class weight for the positive class capped at 20 in every framework (XGBoost
`scale_pos_weight`, weighted loss in Keras, fastai and Flash). TabNet used its balanced sampler.

**6. Why cap the weight at 20?**
The natural weight is about 450 on the credit card data, which makes neural-network training unstable. Using the
same cap everywhere keeps the comparison fair.

**7. What is data leakage and how did you avoid it?**
Using information from the test set during training. I fitted the scaler on the training part only and split before
any model fitting. I also removed duplicate rows (all legitimate) so identical rows cannot land in both parts.

**8. What are V1 to V28?**
PCA components of the original card features, anonymised for privacy. We can rank them (V17, V14, V3, V12, V10 are
the most correlated with fraud) but not interpret them in business terms.

**9. Which framework won and why?**
XGBoost: best F1 on both datasets (0.90 credit card, 0.61 German), shortest code, fastest to train and its trees and
feature importances can be inspected. Tree ensembles are strong on tabular data.

**10. Why did TabNet do badly on credit card data?**
Its balanced sampler pushes predicted fraud probabilities up, so it flagged 115 cases to catch 38 frauds (77 false
alarms). ROC-AUC was still 0.973, so its ranking is fine; the problem is the 0.5 threshold. It also took 458 seconds.

**11. What is the difference between ROC-AUC and F1?**
ROC-AUC measures how well the model ranks fraud above non-fraud across all thresholds. F1 is the balance of
precision and recall at one threshold (0.5 here).

**12. Why is R² negative for TabNet?**
Its predicted probabilities are, on average, further from the true labels than simply predicting the base rate for
everyone, so R² drops below zero.

**13. What is TabNet?**
A neural network for tabular data that uses sequential attention to pick which features to look at in each step. It
gives built-in feature importance. On credit card data it ranked V14, V17 and V10 highest.

**14. What is Lightning Flash and what went wrong with it?**
A high-level wrapper over PyTorch Lightning. It is no longer maintained, so it needed older library versions and a
small compatibility patch (`flash_compat.py`) to run on Python 3.13.

**15. Are the results reliable?**
Directionally yes, precisely no. It is one split and one seed, the credit card test set has only 45 frauds (one
missed fraud moves recall by 2.2 points), and the German test set has 60 bad cases. Cross-validation would be the
next step.

**16. What would you do next?**
Cross-validation, tune the decision threshold per model, try SMOTE, add SHAP explanations for XGBoost, and train the
full 284,807-row Kaggle dataset.

**17. Why is the German data harder?**
Only 1,000 rows and 20 attributes with weak individual signals, so ROC-AUC tops out around 0.79 while credit card
fraud reaches 0.97+.

**18. Something odd you found in the German data?**
Applicants with a "critical account" history have the lowest bad-credit rate (17%). It is a known quirk, so we
avoid reading it as a cause.
