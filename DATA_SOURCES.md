# Phase B data sources

Each dataset must be downloaded manually (no Kaggle/UCI credentials are
configured for this project) and placed under `data/raw/<name>/`. Nothing
under `data/raw/` is committed to git (see `.gitignore`).

| name (loader key) | source | download | place file(s) as |
|---|---|---|---|
| `communities_crime` | UCI | https://archive.ics.uci.edu/dataset/183/communities+and+crime | `data/raw/communities_crime/communities.data` (+ optionally `communities.names`) |
| `credit_card_fraud` | Kaggle `mlg-ulb/creditcardfraud` | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud | `data/raw/credit_card_fraud/creditcard.csv` |
| `nih_chest_xray_sample` | Kaggle `nih-chest-xrays/sample` | https://www.kaggle.com/datasets/nih-chest-xrays/sample | `data/raw/nih_chest_xray_sample/` (extract as-is) — **see note below** |
| `bioassay` | Kaggle `uciml/bioassay-datasets` | https://www.kaggle.com/datasets/uciml/bioassay-datasets | `data/raw/bioassay/*.csv` (this is a collection of many PubChem assay CSVs — see note below) |
| `synthetic_financial_fraud` | Kaggle `umitka/synthetic-financial-fraud-dataset` | https://www.kaggle.com/datasets/umitka/synthetic-financial-fraud-dataset | `data/raw/synthetic_financial_fraud/*.csv` |
| `fraud_detection_transactions` | Kaggle `samayashar/fraud-detection-transactions-dataset` | https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset | `data/raw/fraud_detection_transactions/*.csv` |
| `credit_card_fraud_2023` | Kaggle `nelgiriyewithana/credit-card-fraud-detection-dataset-2023` | https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023 | `data/raw/credit_card_fraud_2023/*.csv` |

Loaders live in `benchmarks/real_datasets.py`. Run
`python -m benchmarks.run_benchmark --dataset real:<name>` once the file is
in place.

## Notes / open items

- **`communities_crime` is a regression dataset.** The loader binarizes it:
  the top `top_pct` (default 5%) of communities by `ViolentCrimesPerPop`
  become the positive ("high-crime") class, matching the project's target
  imbalance range. This is a modeling choice, not the dataset's native
  framing — reconsider `top_pct` if a different ratio is wanted.
- **`bioassay` is a collection, not one file.** Kaggle's `uciml/bioassay-
  datasets` bundles many separate PubChem AID assay CSVs. The loader picks
  the first `*.csv` it finds in the folder and prints which one — drop only
  the specific assay file(s) you want benchmarked in there, or pass
  `filename=` explicitly to `load_bioassay()`.
- **`synthetic_financial_fraud`, `fraud_detection_transactions`,
  `credit_card_fraud_2023`**: schemas weren't verified against the actual
  files (not yet downloaded as of writing), so these use a generic loader
  that auto-detects the label column from common names (`Class`, `isFraud`,
  `is_fraud`, `fraud`, `fraud_bool`, `target`, `label`, `Is Fraudulent`,
  `outcome`) and one-hot encodes low-cardinality categoricals. If the label
  column isn't auto-detected, `load_generic_tabular(name, target_col=...)`
  takes an explicit override — it will raise a clear error listing the
  columns it saw rather than guessing wrong.
- **`nih_chest_xray_sample` is image data**, not tabular. It needs a
  fundamentally different pipeline (image loading + CNN/embedding feature
  extraction) than the rest of this benchmark. No loader is implemented yet
  — flagging this for a scoping decision before building it, rather than
  bolting on an untested image pipeline.
