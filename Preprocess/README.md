\# 📂 Preprocess — Feature Engineering (TimeFix)



This folder contains the \*\*feature engineering pipeline\*\* for the project.  

It reads the raw CSV files from `../data/`, performs aggregation and time-based feature engineering, and outputs the feature tables used by the model.



---



\## 📁 Files



\### `feature\_engineering\_timefix.py`



Main script for \*\*building training and prediction features\*\*.



It:



1\. Reads raw input files from `../data/`:

&nbsp;  - `acct\_transaction.csv`

&nbsp;  - `acct\_alert.csv`

&nbsp;  - `acct\_predict.csv`

2\. Normalizes and cleans the data:

&nbsp;  - Parses dates into integer day indices.

&nbsp;  - Converts transaction time into minute-of-day, 5-min bins, peak/night flags.

&nbsp;  - Compresses categorical columns (`currency\_type`, `channel\_type`, account type flags).

&nbsp;  - Clips transaction amounts with winsorization to reduce extreme values.

3\. Builds a \*\*long-format transaction table\*\* (`tx\_long`) with both payer and payee views.

4\. Aggregates per-account features, including:

&nbsp;  - Transaction counts, sums, basic statistics.

&nbsp;  - Counterparty diversity and entropy.

&nbsp;  - Time-bin entropy and activity patterns.

5\. Constructs:

&nbsp;  - \*\*Training features\*\* (with positive + hard negative accounts)

&nbsp;  - \*\*Prediction features\*\* (for accounts in `acct\_predict.csv`)

6\. Outputs three files to `../data/`:

&nbsp;  - `features\_train.csv` — labeled training features (`acct`, `label`, `is\_unlabeled`, feature columns)

&nbsp;  - `features\_pred.csv` — prediction features (`acct`, feature columns)

&nbsp;  - `features\_meta.json` — metadata (feature list, caps, log-transform info, etc.)



---



\## 🔧 How to run (from project root)



\### 1️⃣ Run feature engineering only



```bash

python -m Preprocess.feature\_engineering\_timefix
```


