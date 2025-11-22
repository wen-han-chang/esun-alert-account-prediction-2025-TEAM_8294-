"""Model training and inference for the FinTech fraud detection task.

This module:
- loads engineered features from the data directory,
- trains a LightGBM model with PU (positive–unlabeled) weighting,
- applies Platt calibration and a secondary ranker,
- outputs the final submit CSV and a detailed score file.
"""
from pathlib import Path

# ================== 路徑設定 ==================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SUBMIT_DIR = BASE_DIR / "submit"
SUBMIT_DIR.mkdir(exist_ok=True)

FEATURES_TRAIN = DATA_DIR / "features_train.csv"
FEATURES_PRED  = DATA_DIR / "features_pred.csv"
FEATURES_META  = DATA_DIR / "features_meta.json"

SUBMIT_CSV     = SUBMIT_DIR / "submit_stack_topk.csv"
SCORES_CSV     = SUBMIT_DIR / "acct_predict_out_stack.csv"

# ================== 參數設定 ==================
ACC0_PUBLIC    = 0.933
RATE           = max(0.001, 1.0 - ACC0_PUBLIC)
K_SHIFT        = 0.0000

BAND           = (0.03, 0.15)
RANKER_SEEDS   = [42, 73, 101, 137]

ALPHA          = 0.65
GAMMA_FIXED    = 0.30
N_FOLDS        = 5
RANDOM_STATE   = 42

# ================== Imports ==================
import json, gc, math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
except:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb


# ================== Utils ==================
def fit_lgbm_classifier(X, y, sample_weight=None, valid=None, seed=42):
    """Train a LightGBM binary classifier with fixed hyperparameters.

    Parameters
    ----------
    X : pandas.DataFrame
        Training feature matrix.
    y : array-like
        Binary labels (0/1).
    sample_weight : array-like, optional
        Per-sample weights. Useful for PU weighting.
    valid : tuple(X_val, y_val), optional
        Validation data used for early stopping. If None, trains without eval set.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    lgb.LGBMClassifier
        Fitted LightGBM model.
    """
    params = dict(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=-1,
        num_leaves=64,
        min_child_samples=40,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1
    )
    model = lgb.LGBMClassifier(**params)
    if valid is not None:
        Xv, yv = valid
        model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=[(Xv, yv)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def k_from_rate(n, rate, k_shift=0.0):
    """Compute the number of top samples given a target rate.

    Parameters
    ----------
    n : int
        Total number of samples.
    rate : float
        Target fraction (e.g. 0.067 for 6.7%).
    k_shift : float, default 0.0
        Small adjustment added to the rate to fine tune k.

    Returns
    -------
    int
        Rounded number of top-k samples.
    """
    rate2 = max(0.0, min(1.0, rate + k_shift))
    return int(round(n * rate2))



# ================== MAIN 模型流程 ==================
def main():
    """Run the full model pipeline on pre-computed features.

    Steps
    -----
    1) Load feature metadata and train/prediction feature tables.
    2) Train a base LightGBM model with PU weighting and K-fold OOF.
    3) Apply Platt calibration on OOF predictions.
    4) Train a ranker on a score band to refine ordering.
    5) Fuse calibrated scores and ranker scores.
    6) Select top-k accounts and save:
       - submit CSV in the `submit/` directory.
       - detailed scores CSV in the `submit/` directory.
    """
    # ---------- 讀取 features ----------
    meta = json.load(open(FEATURES_META, "r", encoding="utf-8"))
    feature_cols = meta.get("feature_cols", [])

    if not feature_cols:
        df_train_head = pd.read_csv(FEATURES_TRAIN, nrows=1)
        feature_cols = [
            c for c in df_train_head.columns if c not in ("acct", "label", "is_unlabeled")
        ]

    train = pd.read_csv(FEATURES_TRAIN)
    pred  = pd.read_csv(FEATURES_PRED)

    # ---------- 對齊欄位 ----------
    for c in feature_cols:
        if c not in pred.columns:
            pred[c] = 0.0
    pred = pred[["acct"] + feature_cols]

    X = train[feature_cols].copy()
    y = train["label"].astype(int).values
    u = train["is_unlabeled"].astype(int).values

    X_te = pred[feature_cols].copy()
    acct_te = pred["acct"].astype(str).values

    print("Train shape:", X.shape, "Test shape:", X_te.shape)

    # ---------- PU 權重 ----------
    w = np.where(y==1, 1.0, GAMMA_FIXED).astype("float32")

    # ---------- KFold ----------
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X), dtype="float32")
    models = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        Xtr, ytr = X.iloc[tr_idx], y[tr_idx]
        Xva, yva = X.iloc[va_idx], y[va_idx]
        wtr = w[tr_idx]

        model = fit_lgbm_classifier(Xtr, ytr, sample_weight=wtr, valid=(Xva, yva),
                                    seed=RANDOM_STATE+fold)
        models.append(model)
        oof[va_idx] = model.predict_proba(Xva)[:,1]
        print(f"[FOLD {fold}] AUC={roc_auc_score(yva, oof[va_idx]):.5f}")

    print(f"[OOF] AUC={roc_auc_score(y, oof):.5f}")

    # ---------- Platt 校準 ----------
    platt = LogisticRegression(max_iter=1000)
    platt.fit(oof.reshape(-1,1), y)
    meta_cal_oof = platt.predict_proba(oof.reshape(-1,1))[:,1]

    pred_raw_list = [m.predict_proba(X_te)[:,1] for m in models]
    meta_raw_te = np.mean(pred_raw_list, axis=0)
    meta_cal_te = platt.predict_proba(meta_raw_te.reshape(-1,1))[:,1]

    # ---------- Ranker ----------
    q_lo, q_hi = np.quantile(oof, BAND[0]), np.quantile(oof, BAND[1])
    band_mask = (oof >= q_lo) & (oof <= q_hi)

    X_band = X.loc[band_mask].reset_index(drop=True)
    y_band = y[band_mask]

    ranker_models = [
        fit_lgbm_classifier(X_band, y_band, seed=s)
        for s in RANKER_SEEDS
    ]
    rank_score_te = np.mean(
        [m.predict_proba(X_te)[:,1] for m in ranker_models], axis=0
    )

    # ---------- Final Score ----------
    final_te = ALPHA * meta_cal_te + (1 - ALPHA) * rank_score_te

    # ---------- 產出 submit ----------
    n_test = len(final_te)
    k = k_from_rate(n_test, RATE, K_SHIFT)
    order = np.argsort(-final_te)
    topk_idx = set(order[:k])

    submit = pd.DataFrame({
        "acct": acct_te,
        "predict": [1 if i in topk_idx else 0 for i in range(n_test)]
    })
    submit.to_csv(SUBMIT_CSV, index=False, encoding="utf-8-sig")

    # ---------- Debug score ----------
    scores = pd.DataFrame({
        "acct": acct_te,
        "final_score": final_te,
        "meta_cal": meta_cal_te,
        "rank_score": rank_score_te,
    }).sort_values("final_score", ascending=False)

    scores.to_csv(SCORES_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved submit → {SUBMIT_CSV}")
    print(f"[OK] Saved scores → {SCORES_CSV}")


# ================== Public rerank（可選） ==================
def public_rerank():
    """Placeholder for public-leaderboard-specific reranking logic.

    This function is intentionally left unimplemented. Implement if
    you need a separate reranking strategy for public leaderboard tuning.
    """
    print("[INFO] public_rerank() 尚未實作，可視需要加回")


# ================== 入口 ==================
if __name__ == "__main__":
    main()
