## 🏆 比賽成績

本專案在 **玉山銀行 2025 Alert Account Prediction** 競賽中獲得：

🎯 **第 36 名 / 790 隊（前 4.5%）**

- 模型：PU-Learning + RankStack + 時間修正特徵 (TimeFix)
- 最佳提交：`submit_public_rerank_k320.csv`
- Public Leaderboard 排名：Top 36
- 參賽隊伍：TEAM_8294

---

# 🔍 File Description — 每個檔案的功能

## 📌 `feature_engineering_timefix.ipynb`
**用途：特徵工程（含時間修正 TimeFix）**

- 讀取官方原始資料（交易資料 + alert 清單 + pred 名單）
- 以 **帳號 acct** 為單位聚合：
  - 金額統計：sum / mean / std / median / max / min
  - 交易次數、唯一對手數、集中度
- **時間週期特徵**（TimeFix）：
  - hour / weekday
  - 短時間連續交易
  - 跨日行為、交易密度
- 將每個 alert 對應的交易聚合成 alert-level 特徵
- 輸出：
  - `features_train.csv`
  - `features_pred.csv`
  - `features_meta.json`

---

## 📌 `train_rankstack_timefix.ipynb`
**用途：PU-Learning + RankStack 模型訓練與分數融合**

### 主要流程：

### 1. Meta Model（Logistic Regression）
- 為每個 acct 給 baseline 機率 `meta_cal`

### 2. BAND 過濾（只使用中間機率區間訓練 ranker）
- 範例區間：`(0.03, 0.15)`

### 3. Ranker（LightGBM）
- Stratified K-Fold
- 多 random seed bagging
- 產生 `rank_score`

### 4. Stacking 融合

```python
final_score = ALPHA * meta_cal + (1 - ALPHA) * rank_score
````

### 5. Top-K 選取

* Public ACC0 ≈ 0.933 → 陽性比例 ≈ **6.7%**
* 輸出：

  * `submit_stack_topk.csv`
  * `acct_predict_out_stack.csv`

