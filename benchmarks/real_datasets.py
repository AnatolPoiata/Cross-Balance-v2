"""Loaders for the Phase B real-world datasets.

Every dataset must be manually downloaded into ``data/raw/<name>/`` first —
see ``DATA_SOURCES.md`` for exact download links and expected file
placement. Loaders raise a clear, actionable error (not a guess) when the
expected file or a target column can't be found.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .datasets import PROJECT_ROOT

RAW_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_TARGET_CANDIDATES = [
    "class", "target", "label", "is_fraud", "isfraud", "fraud",
    "fraud_bool", "is fraudulent", "outcome", "y",
]


def _missing_file_error(path, dataset_name):
    return FileNotFoundError(
        f"Expected data for '{dataset_name}' at {path}, but it doesn't exist.\n"
        f"See DATA_SOURCES.md for the download link and where to place the file."
    )


def _encode_target(y_raw, verbose_name="target"):
    """Map an arbitrary 2-class target to {0, 1}, with 1 = the minority
    (positive-of-interest) class. Returns an int array.
    """
    if y_raw.dtype == bool:
        y = y_raw.astype(int).to_numpy()
        return y

    uniques = y_raw.unique()
    if len(uniques) != 2:
        raise ValueError(
            f"Expected a binary {verbose_name} column, found {len(uniques)} "
            f"unique values: {sorted(map(str, uniques))[:10]}"
        )
    if set(uniques) == {0, 1}:
        return y_raw.astype(int).to_numpy()

    counts = y_raw.value_counts()
    minority_value = counts.idxmin()
    majority_value = counts.idxmax()
    print(f"  [_encode_target] mapping {verbose_name}=={minority_value!r} -> 1 (minority, n={counts.min()}), "
          f"{verbose_name}=={majority_value!r} -> 0 (n={counts.max()})")
    return (y_raw == minority_value).astype(int).to_numpy()


def _detect_target_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    for cand in candidates:
        matches = [c for c in df.columns if cand in c.lower()]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(
        "Could not auto-detect the target column. Pass target_col= explicitly.\n"
        f"Columns available: {list(df.columns)}"
    )


def _drop_id_like_columns(df):
    # Only integer/string/object columns can be "ID-like" — continuous float
    # features (e.g. transaction amount) are naturally all-unique too, and
    # are exactly the kind of signal we must not discard.
    candidate_cols = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    id_like = [c for c in candidate_cols if df[c].nunique(dropna=False) == len(df)]
    if id_like:
        print(f"  [_drop_id_like_columns] dropping ID-like columns (all-unique values): {id_like}")
    return df.drop(columns=id_like)


def _encode_features(X_df, max_onehot_cardinality=20):
    X_df = _drop_id_like_columns(X_df)
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    low_card = [c for c in categorical_cols if X_df[c].nunique(dropna=True) <= max_onehot_cardinality]
    dropped = set(categorical_cols) - set(low_card)
    if dropped:
        print(f"  [_encode_features] dropping high-cardinality non-numeric columns: {sorted(dropped)}")

    parts = [X_df[numeric_cols].apply(lambda s: s.fillna(s.median()))]
    if low_card:
        parts.append(pd.get_dummies(X_df[low_card].astype("category"), dummy_na=False))
    X_encoded = pd.concat(parts, axis=1)
    return X_encoded.to_numpy(dtype=float)


def load_generic_tabular(name, target_col=None, filename=None,
                          candidate_target_names=DEFAULT_TARGET_CANDIDATES):
    """Generic loader for a single-CSV tabular binary-classification dataset
    whose exact schema isn't hardcoded — auto-detects the label column and
    one-hot encodes low-cardinality categoricals.
    """
    dir_path = RAW_DIR / name
    if filename is not None:
        path = dir_path / filename
        if not path.exists():
            raise _missing_file_error(path, name)
    else:
        # Exclude macOS AppleDouble sidecar files (e.g. "._data.csv"), which
        # this project's network volume generates alongside every file.
        csvs = sorted(p for p in dir_path.glob("*.csv") if not p.name.startswith("._"))
        if not csvs:
            raise _missing_file_error(dir_path / "*.csv", name)
        path = csvs[0]
        if len(csvs) > 1:
            print(f"  [load_generic_tabular:{name}] multiple CSVs found, using {path.name} "
                  f"(others: {[p.name for p in csvs[1:]]})")

    df = pd.read_csv(path)
    if target_col is None:
        target_col = _detect_target_column(df, candidate_target_names)
    y = _encode_target(df[target_col], verbose_name=target_col)
    X = _encode_features(df.drop(columns=[target_col]))
    return X, y


def load_credit_card_fraud():
    """Kaggle mlg-ulb/creditcardfraud: Time, V1-V28 (PCA features), Amount, Class."""
    path = RAW_DIR / "credit_card_fraud" / "creditcard.csv"
    if not path.exists():
        raise _missing_file_error(path, "credit_card_fraud")
    df = pd.read_csv(path)
    y = df["Class"].astype(int).to_numpy()
    X = df.drop(columns=["Class"]).to_numpy(dtype=float)
    return X, y


def load_communities_crime(top_pct=0.05):
    """UCI Communities and Crime: binarized by taking the top ``top_pct`` of
    communities by ViolentCrimesPerPop as the positive ("high-crime") class.
    First 5 columns (state, county, community, communityname, fold) are
    non-predictive per the dataset documentation and are dropped.
    """
    path = RAW_DIR / "communities_crime" / "communities.data"
    if not path.exists():
        raise _missing_file_error(path, "communities_crime")

    df = pd.read_csv(path, header=None, na_values="?")
    df = df.drop(columns=[0, 1, 2, 3, 4])  # non-predictive identifier columns
    target_continuous = df.iloc[:, -1]
    features = df.iloc[:, :-1]

    # Drop columns missing in >50% of rows, median-impute the rest.
    missing_frac = features.isna().mean()
    features = features.drop(columns=missing_frac[missing_frac > 0.5].index)
    features = features.apply(lambda s: s.fillna(s.median()))

    threshold = np.quantile(target_continuous, 1 - top_pct)
    y = (target_continuous >= threshold).astype(int).to_numpy()
    X = features.to_numpy(dtype=float)
    print(f"  [load_communities_crime] top_pct={top_pct}, threshold={threshold:.4f}, "
          f"positive_count={int(y.sum())}/{len(y)}")
    return X, y


def load_bioassay(filename=None):
    """Kaggle uciml/bioassay-datasets is a collection of many PubChem assay
    CSVs. Uses the generic tabular loader against a single chosen file.
    """
    return load_generic_tabular("bioassay", filename=filename,
                                 candidate_target_names=["outcome", "activity", "active", "class"] +
                                 DEFAULT_TARGET_CANDIDATES)


def load_synthetic_financial_fraud():
    return load_generic_tabular("synthetic_financial_fraud")


def load_fraud_detection_transactions():
    return load_generic_tabular("fraud_detection_transactions")


def load_credit_card_fraud_2023():
    return load_generic_tabular("credit_card_fraud_2023")


REAL_DATASETS = {
    "communities_crime": load_communities_crime,
    "credit_card_fraud": load_credit_card_fraud,
    "bioassay": load_bioassay,
    "synthetic_financial_fraud": load_synthetic_financial_fraud,
    "fraud_detection_transactions": load_fraud_detection_transactions,
    "credit_card_fraud_2023": load_credit_card_fraud_2023,
    # nih_chest_xray_sample intentionally omitted: image data needs a
    # separate embedding pipeline, see DATA_SOURCES.md.
}
