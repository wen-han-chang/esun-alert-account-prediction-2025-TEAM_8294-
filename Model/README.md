
---

## 🤖 `Model/README.md`

```markdown
# 📂 Model — Alert Account Prediction Model

This folder contains the **training and inference pipeline** for the competition model.  
It consumes features produced by `Preprocess/feature_engineering_timefix.py` and outputs submit-ready prediction files.

---

## 📁 Files

### `model.py`

Main script for **training the model and generating predictions**.

It:

1. Reads feature files from `../data/`:
   - `features_train.csv`
   - `features_pred.csv`
   - `features_meta.json`
2. Loads feature metadata:
   - Uses `feature_cols` from `features_meta.json`
   - If missing, falls back to auto-detecting feature columns from `features_train.csv`.
3. Trains a **PU-learning style LightGBM classifier**:
   - Positive samples: labeled alerts (`label = 1`)
   - Unlabeled samples: non-alert accounts (`label = 0`, `is_unlabeled = 1`)
   - Applies class weights via `GAMMA_FIXED` for unlabeled data.
   - Uses Stratified K-Fold (`N_FOLDS`) with out-of-fold (OOF) predictions.
4. Applies **Platt scaling** (Logistic Regression) on OOF scores:
   - Calibrates raw probabilities to obtain `meta_cal`.
5. Trains a **Ranker model** (LightGBM) on a middle probability band (`BAND`):
   - Trained only on samples with OOF scores within the specified quantile range.
   - Ranker outputs `rank_score` for test accounts.
6. Combines meta scores and rank scores:
   - Final score:
     ```python
     final_score = ALPHA * meta_cal + (1 - ALPHA) * rank_score
     ```
7. Selects Top-K accounts based on `final_score`:
   - `K` is determined from `RATE` (derived from public ACC0)
8. Outputs two CSV files:

   - `../submit/submit_stack_topk.csv`
     - Final submission file with:
       - `acct`
       - `predict` (0/1)
   - `../submit/acct_predict_out_stack.csv`
     - Debug / analysis file with:
       - `acct`
       - `final_score`
       - `meta_cal`
       - `rank_score`

---

## 🔧 How to run (from project root)

### 1️⃣ Run model only (assumes features already exist)

```bash
python -m Model.model
```
