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
| `default_credit_card_clients` | UCI: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients | `default of credit card clients.xls` | ~3.5:1 (22.1% minority) |
| `mammographic` | UCI: https://archive.ics.uci.edu/dataset/161/mammographic+mass | `mammographic_masses.data`, `.names` | **near-balanced** (46.3% minority) |
| `mushrooms` | UCI: https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset | `secondary_data.csv` (+ `primary_data.csv`, unused) | **near-balanced** (44.5% minority) |
| `bank_marketing` | UCI: https://archive.ics.uci.edu/dataset/222/bank+marketing | `bank-additional-full.csv` (+ `bank-additional.csv`, unused 10% subset) | ~7.9:1 (11.3% minority) |

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
- **`mammographic` and `mushrooms` are both close to balanced** (46.3% and
  44.5% minority respectively) despite being framed as classification
  benchmarks — included for domain diversity (medical diagnosis, biological
  taxonomy) rather than as core imbalance tests; treat them like
  `credit_card_fraud_2023` as control/diversity points.
- **`mammographic` drops the `BI-RADS` column** — the dataset's own `.names`
  file documents it as "non-predictive": it's the radiologist's own
  pre-assessment, so using it as a model feature would leak the near-answer.
  Uses Age/Shape/Margin/Density as features (missing values median-imputed).
- **`default_credit_card_clients`** is an `.xls` file with a two-row header
  (generic `X1..X23/Y` labels, then the real column names) — the loader
  reads with `header=1` to skip the generic row. Requires the `xlrd` package
  (`pip install -e ".[bench]"`).
- **`mushrooms` uses `secondary_data.csv`**, not `primary_data.csv` — the
  latter stores feature *ranges* (e.g. `"[10, 20]"` for cap diameter) rather
  than per-instance values, and isn't usable as numeric/categorical features
  directly. Both files are semicolon-delimited.
- **`bank_marketing` uses `bank-additional-full.csv`**, not the 10%-subset
  `bank-additional.csv`. Semicolon-delimited with quoted string fields.

## Hold-out collection (added for v3, first loaded after the v3 design freeze)

These 17 datasets were not used at any point while developing v1-v3 (the
v3 development grid used only the datasets above). They are evaluated once,
with the frozen v0.2 defaults, by `benchmarks/run_v3.py` (`HOLDOUT_DATASETS`
in `real_datasets.py`).

| loader key | source | file | rows | minority |
|---|---|---|---|---|
| `creditcard_ulb` | ULB / Worldline, Kaggle `mlg-ulb/creditcardfraud` (Dal Pozzolo et al., 2015) | `data/raw/credit_card/creditcard.csv` | 284,807 | 492 (0.17%) — **run at full size, native prevalence** |
| `rr_*` (16 sets) | Park & Im (2024), Mendeley Data, doi:10.17632/pb3jd7vz9z.1 (KEEL/UCI-derived; paper: Park, Lee & Im, ESWA 247, 123274) | `data/raw/Experimental Dataset for Imbalanced Classification Application of Relabeling & Ranking Algorithm/[Cate|Conti|Mixed] <name>.csv` | 214 – 18,916 | IR 10 – 42 |

The `rr_*` files carry a leading row-index column (named `""`, `Unnamed: 0`
or `0`; its first value is corrupted to `Fa` in `Led.csv`), which the loader
drops; the label is the last column (minority mapped to 1). Binary `Tr`/`Fa`
and other categorical columns are one-hot encoded.

## Notes added during v3

- **Ailerons' positive share is 9.66%, not 5%, because of ties.** `Goal`
  has only 35 distinct values; the 95th-percentile value (-0.0004) is shared
  by 1,084 rows, and the loader's rule `Goal >= quantile(0.95)` keeps all
  ties (1,328 positives / 13,750).
- **The 200-positive floor of `stratified_subsample()` raises prevalence**
  for PaySim (0.13% -> 0.40% in the 50,000-row sample) and the Sparkov
  `credit_card_fraud` set (0.52%, close to native). `creditcard_ulb` is
  therefore run unsubsampled.
