# Phase B data sources

Each dataset must be downloaded manually (no Kaggle/UCI credentials are
configured for this project) and placed under `data/raw/<name>/`. Nothing
under `data/raw/` is committed to git (see `.gitignore`).

`bioassay` and `nih_chest_xray_sample` are **excluded per project decision**
(multi-file collection needing a per-assay choice; image data needing a
separate embedding pipeline, respectively) — no loaders exist for them.

| name (loader key) | source | actual file(s) in `data/raw/<name>/` | native imbalance |
|---|---|---|---|
| `communities_crime` | UCI: https://archive.ics.uci.edu/dataset/183/communities+and+crime | `communities.data`, `communities.names` | regression — binarized, top 5% `ViolentCrimesPerPop` |
| `credit_card_fraud` | Kaggle `kartik2112/fraud-detection` (not mlg-ulb — see note) | `fraudTrain.csv`, `fraudTest.csv` | ~172:1 (0.58% fraud) |
| `credit_card_fraud_2023` | Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023` | `creditcard_2023.csv` | **50:50 — not imbalanced**, see note |
| `synthetic_financial_fraud` | Kaggle `umitka/synthetic-financial-fraud-dataset` (PaySim) | `PS_20174392719_1491204439457_log.csv` | ~773:1 (0.13% fraud) |
| `fraud_detection_transactions` | Kaggle `samayashar/fraud-detection-transactions-dataset` | `synthetic_fraud_dataset.csv` | ~2.1:1 (32% fraud) — mild |
| `ailerons` | https://github.com/antoniopedropi/Datasets-ImbalancedRegression | `ailerons.csv` | regression — binarized, top 5% `Goal` |
| `segment0` | KEEL: https://sci2s.ugr.es/keel/imbalanced.php | `segment0.dat` (`data/raw/segemnt0/`, note the folder typo) | ~6:1 (14.3% minority) |
| `shuttle_2_vs_5` | KEEL: https://sci2s.ugr.es/keel/dataset.php?cod=1342 | `shuttle-2_vs_5.dat` | ~67:1 (1.5% minority) |

Loaders live in `benchmarks/real_datasets.py`. Run
`python -m benchmarks.run_benchmark --dataset real:<name>` (or `real:all`).
Datasets larger than `--max-samples` (default 50,000) are stratified-
subsampled, preserving the original class ratio (with a floor on minority
count so extreme-imbalance sets like PaySim still carry enough positives
for k-fold CV) — see `stratified_subsample()` in `real_datasets.py`.

## Notes / open items

- **`credit_card_fraud` is not the mlg-ulb dataset originally planned.**
  The file placed here is Kaggle `kartik2112/fraud-detection`
  (`fraudTrain.csv`/`fraudTest.csv`: simulated transactions with cardholder
  PII — name, address, DOB, card number, merchant, job title, lat/long).
  The loader drops direct identifiers and free-text/high-cardinality
  columns, keeps `amt`, `category`, `gender`, geography, `unix_time`,
  `city_pop`, and concatenates both files. Target: `is_fraud`.
- **`credit_card_fraud_2023` is a perfectly balanced 50:50 dataset**
  (284,315 fraud / 284,315 non-fraud) — not actually imbalanced, despite
  the name. Still included per the original dataset list, but it isn't a
  meaningful imbalance-handling comparison on its own; treat its results as
  a "control" data point rather than part of the core imbalance story.
- **`communities_crime` / `ailerons` are regression datasets.** Both are
  binarized by taking the top `top_pct` (default 5%) of rows by the
  continuous target as the positive class — a modeling choice, not the
  dataset's native framing.
- **`segment0` / `shuttle_2_vs_5`** are parsed from KEEL's `.dat` format
  (`@relation`/`@attribute`/`@data` header, comma-separated rows, last
  column `Class` with `{positive, negative}` labels — `positive` is the
  minority class by KEEL convention, mapped to `1`).
- **`synthetic_financial_fraud`, `fraud_detection_transactions`,
  `credit_card_fraud_2023`** use the generic auto-detecting tabular loader
  (`load_generic_tabular`): label column found via common-name matching
  (`Class`, `isFraud`, `Fraud_Label`, etc.), ID-like columns dropped,
  low-cardinality categoricals one-hot encoded, high-cardinality
  categoricals (raw timestamps, account/user IDs) dropped. Verified against
  the actual files, not guessed.
