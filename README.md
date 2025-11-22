# 🏦 FinTech Alert Account Prediction Pipeline  
**Python 3.13.5 | PU-Learning + RankStack | TimeFix Feature Engineering**

本專案為玉山銀行 2025 **Alert Account Prediction** 競賽的完整可重現 Pipeline，  
包含資料前處理、特徵工程、模型訓練、預測與輸出 submit 的全流程。

---

# 📂 Project Structure

```
.
├── data/                     # 原始資料 + 產生的 features
│   ├── acct_alert.csv
│   ├── acct_predict.csv
│   ├── acct_transaction.csv
│   ├── features_train.csv    # preprocess 產生
│   ├── features_pred.csv     # preprocess 產生
│   └── features_meta.json    # preprocess 產生
│
├── Preprocess/
│   └── feature_engineering_timefix.py  # 特徵工程（TimeFix + PU-friendly）
│
├── Model/
│   └── model.py              # PU-Learning + RankStack + LightGBM pipeline
│
├── submit/
│   └── submit_stack_topk.csv # 模型輸出
│
├── main.py                   # 執行管線入口（先 preprocess → 再 model）
├── requirements.txt          # 套件需求
└── README.md                 # 專案說明（本檔）
```

---

# 🚀 Pipeline Overview

整體流程如下：

```
Raw Data (data/*.csv)
        │
        ▼
[1] Preprocess (feature_engineering_timefix.py)
        ├─ clean & normalize
        ├─ TimeFix time-window aggregation
        ├─ Hard Negative mining (PU-learning)
        ├─ channel / currency wide features
        ├─ entropy / activity features
        └→ features_train.csv, features_pred.csv, features_meta.json
        │
        ▼
[2] Model (model.py)
        ├─ LightGBM PU classifier (meta model)
        ├─ Platt scaling (Logistic Regression)
        ├─ Middle-band Ranker (multi-seed LightGBM)
        ├─ Score fusion (RankStack)
        └→ submit_stack_topk.csv, acct_predict_out_stack.csv
        │
        ▼
[3] Submit
        ✔ 檔案格式符合競賽要求
```

---

# 🧩 Features Included (TimeFix)

Preprocess 會替每個 acct 建立行為特徵，包括：

### ✔ 基礎統計  
- tx_cnt / active_days  
- amt_in_sum / amt_out_sum  
- abs(amount) mean/std/max  
- uniq counterparty

### ✔ TimeFix 時間修正特徵  
- 5-min bin activity entropy  
- peak / night ratio  
- min-of-day distribution  
- recent-window transactions (1–60 天)

### ✔ 類別特徵 (wide encoding)  
- channel_type
- currency_bucket

### ✔ PU-learning Hard Negatives  
- 活躍度高 / 噪音少的未標帳號作為 U 樣本  
- 適用於競賽的 Positive-Unlabeled 監督情境

所有特徵與設定會寫入：

```
data/features_meta.json
```

---

# 🤖 Model Architecture（PU-Learning + RankStack）

模型流程包含：

### **1. Meta Model (LightGBM)**
- stratified K-fold
- PU-learning weight
- early stopping
- 產生 baseline probability

### **2. Platt Scaling**
- Logistic Regression 對 meta score 校準  
- 輸出 `meta_cal`

### **3. Middle-band Ranker**
只訓練中間機率區間 `(0.03, 0.15)`，避免雜訊。

使用：
- 多 SEED bagging（42, 73, 101, 137）
- 單層 LightGBM ranker

### **4. Stacking Ensemble**

```
final_score = ALPHA * meta_cal + (1 - ALPHA) * rank_score
```

### **5. Top-K 選取**
根據 Public ACC0 設定陽性比例：

```
RATE = 1 - ACC0_PUBLIC
```

排序後取前 K：

```
predict = 1 if rank in top-K else 0
```

---

# 📦 Installation

### 1. 使用 Python 3.13.5 建立環境

```bash
python3 -m venv finenv
source finenv/bin/activate
```

### 2. 安裝必要套件

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# ▶️ Run the Entire Pipeline

只需要一行指令：

```bash
python main.py
```

流程會自動：

1. 執行 Preprocess  
2. 產生 features  
3. 執行模型  
4. 產生 submit 檔案  

輸出位置：

```
submit/submit_stack_topk.csv
submit/acct_predict_out_stack.csv
```

---

# 🗂 Folder Descriptions

| Folder / File | Description |
|---------------|-------------|
| **data/** | 原始資料與 preprocess 產生的特徵檔 |
| **Preprocess/** | TimeFix 特徵工程腳本 |
| **Model/** | RankStack / LightGBM 模型 |
| **submit/** | 最終預測 CSV |
| **main.py** | Pipeline 入口（preprocess → model） |
| **requirements.txt** | 套件需求 |
| **README.md** | 專案說明文件 |

---

# 🏁 Competition Result

本專案於 **玉山銀行 2025 Alert Account Prediction** 競賽取得：

🎯 **第 36 名 / 790 隊（前 4.5%）**

- 模型：PU-Learning + RankStack + TimeFix  
- Public Leaderboard：Top 36  
- Team：TEAM_8294  

---

# 📬 Contact
對專案架構、模型或特徵工程有任何疑問，歡迎提出！

