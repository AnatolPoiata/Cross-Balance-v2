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
    "fraud_bool", "fraud_label", "is fraudulent", "outcome", "y",
]


def stratified_subsample(X, y, max_samples, random_state=42, min_minority=200):
    """Cap dataset size while preserving the original class ratio (subject
    to a floor on minority count, so very rare-event datasets still have
    enough positives for k-fold CV to be meaningful).
    """
    n = len(y)
    if n <= max_samples:
        return X, y

    rng = np.random.RandomState(random_state)
    idx_min = np.where(y == 1)[0]
    idx_maj = np.where(y == 0)[0]

    frac = max_samples / n
    n_min_target = min(len(idx_min), max(min_minority, int(round(len(idx_min) * frac))))
    n_maj_target = min(len(idx_maj), max_samples - n_min_target)

    sel_min = rng.choice(idx_min, size=n_min_target, replace=False)
    sel_maj = rng.choice(idx_maj, size=n_maj_target, replace=False)
    idx = np.concatenate([sel_min, sel_maj])
    rng.shuffle(idx)
    print(f"  [stratified_subsample] {n} -> {len(idx)} rows "
          f"(minority {len(idx_min)} -> {n_min_target}, majority {len(idx_maj)} -> {n_maj_target})")
    return X[idx], y[idx]


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


_CCF_PII_DROP_COLS = [
    "Unnamed: 0", "cc_num", "first", "last", "street", "dob", "trans_num",
    "merchant", "job", "trans_date_trans_time",
]


def load_credit_card_fraud():
    """Kaggle kartik2112/fraud-detection (``fraudTrain.csv`` + ``fraudTest.csv``):
    simulated card-present/not-present transactions with cardholder PII.
    Drops direct identifiers and free-text/high-cardinality columns (name,
    address, DOB, card number, merchant name, job title, raw timestamp);
    keeps amount, category, gender, geography, unix_time, city_pop.
    Target: ``is_fraud``.
    """
    train_path = RAW_DIR / "credit_card_fraud" / "fraudTrain.csv"
    test_path = RAW_DIR / "credit_card_fraud" / "fraudTest.csv"
    if not train_path.exists():
        raise _missing_file_error(train_path, "credit_card_fraud")

    parts = [pd.read_csv(train_path)]
    if test_path.exists():
        parts.append(pd.read_csv(test_path))
    df = pd.concat(parts, ignore_index=True)

    y = df["is_fraud"].astype(int).to_numpy()
    X_df = df.drop(columns=["is_fraud"] + [c for c in _CCF_PII_DROP_COLS if c in df.columns])
    X = _encode_features(X_df)
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


def load_synthetic_financial_fraud():
    return load_generic_tabular("synthetic_financial_fraud")


def load_fraud_detection_transactions():
    return load_generic_tabular("fraud_detection_transactions")


def load_credit_card_fraud_2023():
    """Kaggle nelgiriyewithana/credit-card-fraud-detection-dataset-2023:
    note this dataset is a perfectly balanced 50:50 synthetic set (not
    imbalanced) — included per the project's dataset list, but it isn't a
    meaningful imbalance-handling comparison on its own.
    """
    return load_generic_tabular("credit_card_fraud_2023")


def load_ailerons(top_pct=0.05):
    """Aircraft aileron-control regression dataset (KEEL/imbalanced-regression
    collection). Binarized by taking the top ``top_pct`` of rows by the
    ``Goal`` target as the positive (rare, high-control-demand) class —
    same convention as ``load_communities_crime``.
    """
    path = RAW_DIR / "ailerons" / "ailerons.csv"
    if not path.exists():
        raise _missing_file_error(path, "ailerons")

    df = pd.read_csv(path)
    target_continuous = df["Goal"]
    features = df.drop(columns=["Goal"])

    threshold = np.quantile(target_continuous, 1 - top_pct)
    y = (target_continuous >= threshold).astype(int).to_numpy()
    X = features.to_numpy(dtype=float)
    print(f"  [load_ailerons] top_pct={top_pct}, threshold={threshold:.6f}, "
          f"positive_count={int(y.sum())}/{len(y)}")
    return X, y


def _parse_keel_dat(path):
    """Parse a KEEL-format .dat file (@relation/@attribute/@data header,
    comma-separated rows, class as the last column with {positive,negative}
    labels — positive is the minority class by KEEL convention).
    """
    with open(path) as f:
        lines = f.readlines()

    data_start = next(i for i, line in enumerate(lines) if line.strip().lower() == "@data") + 1
    attr_names = [line.split()[1] for line in lines if line.strip().lower().startswith("@attribute")]

    rows = [line.strip().split(",") for line in lines[data_start:] if line.strip()]
    df = pd.DataFrame(rows, columns=attr_names)

    class_col = attr_names[-1]
    y_raw = df[class_col].str.strip().str.lower()
    y = (y_raw == "positive").astype(int).to_numpy()
    X = df.drop(columns=[class_col]).apply(pd.to_numeric).to_numpy(dtype=float)
    return X, y


def load_segment0():
    """KEEL segment0: image-segmentation, one-vs-rest, ~14.3% minority."""
    path = RAW_DIR / "segemnt0" / "segment0.dat"
    if not path.exists():
        raise _missing_file_error(path, "segment0")
    return _parse_keel_dat(path)


def load_shuttle_2_vs_5():
    """KEEL shuttle-2_vs_5: extreme imbalance, ~1.5% minority (66:1)."""
    path = RAW_DIR / "shuttle-2" / "shuttle-2_vs_5.dat"
    if not path.exists():
        raise _missing_file_error(path, "shuttle-2")
    return _parse_keel_dat(path)


def load_default_credit_card_clients():
    """UCI Default of Credit Card Clients (30,000 rows). The source .xls has
    a two-row header (generic X1..X23/Y labels, then real names) -- read
    with header=1 to get the real names. Drops the ID column. Target:
    ``default payment next month``. Mild imbalance (~22% minority).
    """
    path = RAW_DIR / "default_credit_card_clients" / "default of credit card clients.xls"
    if not path.exists():
        raise _missing_file_error(path, "default_credit_card_clients")
    df = pd.read_excel(path, header=1)
    y = _encode_target(df["default payment next month"], verbose_name="default payment next month")
    X = _encode_features(df.drop(columns=["default payment next month"]))
    return X, y


def load_mammographic(drop_birads=True):
    """UCI Mammographic Mass (961 rows): BI-RADS, Age, Shape, Margin,
    Density, Severity (target). BI-RADS is documented by the dataset itself
    as "non-predictive" (it's the radiologist's own assessment, using it as
    a feature would leak the near-answer) and is dropped by default.
    Missing values ("?") are median-imputed. Near-balanced (~46% minority).
    """
    path = RAW_DIR / "mammographic" / "mammographic_masses.data"
    if not path.exists():
        raise _missing_file_error(path, "mammographic")
    df = pd.read_csv(path, header=None, na_values="?",
                      names=["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"])
    if drop_birads:
        df = df.drop(columns=["BI-RADS"])
    y = df["Severity"].astype(int).to_numpy()
    features = df.drop(columns=["Severity"]).apply(lambda s: s.fillna(s.median()))
    return features.to_numpy(dtype=float), y


def load_mushrooms():
    """UCI Secondary Mushroom Dataset (61,069 rows, semicolon-delimited).
    Uses secondary_data.csv (real feature values); primary_data.csv stores
    ranges/lists (e.g. "[10, 20]") and isn't directly usable as features.
    Target: ``class`` (edible/poisonous) -- note this dataset is close to
    balanced (~44.5%/55.5%), not strongly imbalanced.
    """
    path = RAW_DIR / "mushrooms" / "secondary_data.csv"
    if not path.exists():
        raise _missing_file_error(path, "mushrooms")
    df = pd.read_csv(path, sep=";")
    y = _encode_target(df["class"], verbose_name="class")
    X = _encode_features(df.drop(columns=["class"]))
    return X, y


def load_bank_marketing():
    """UCI Bank Marketing (41,188 rows, semicolon-delimited, quoted strings).
    Uses bank-additional-full.csv (the full set; bank-additional.csv is a
    10% subsample of the same data). Target: ``y`` (term deposit
    subscription). Moderate imbalance (~11.3% minority).
    """
    path = RAW_DIR / "bank_marketing" / "bank-additional-full.csv"
    if not path.exists():
        raise _missing_file_error(path, "bank_marketing")
    df = pd.read_csv(path, sep=";")
    y = _encode_target(df["y"], verbose_name="y")
    X = _encode_features(df.drop(columns=["y"]))
    return X, y


# --- Hold-out collection added after the v3 design freeze -----------------
# Park & Im (2024) "Experimental dataset for imbalanced classification:
# application of relabeling & raking algorithm", Mendeley Data,
# doi:10.17632/pb3jd7vz9z.1 -- 16 KEEL/UCI-derived sets grouped by feature
# type ([Cate]/[Conti]/[Mixed]). Every file carries a leading row-index
# column (named "", "Unnamed: 0" or "0") and the label in the last column.
RR_DIR = RAW_DIR / "Experimental Dataset for Imbalanced Classification Application of Relabeling & Ranking Algorithm"
RR_FILES = {
    "rr_led": "[Cate] Led.csv", "rr_phoss": "[Cate] PhosS.csv", "rr_boundary": "[Cate] boundary.csv",
    "rr_cam": "[Cate] cam.csv", "rr_car_good": "[Cate] car-good.csv",
    "rr_compustat": "[Conti] compustat.csv", "rr_glass4": "[Conti] glass4.csv",
    "rr_mammography": "[Conti] mammography.csv", "rr_oil": "[Conti] oil.csv",
    "rr_winequality_red_8vs6": "[Conti] winequality-red8vs6.csv", "rr_yeast_2vs8": "[Conti] yeast-2vs8.csv",
    "rr_abalone_17vs7_10": "[Mixed] abalone-17vs7-8-9-10.csv", "rr_flare_f": "[Mixed] flare-F.csv",
    "rr_hypo": "[Mixed] hypo.csv", "rr_sick": "[Mixed] sick.csv", "rr_vowel0": "[Mixed] vowel0.csv",
}


def load_relabeling_raking(name):
    path = RR_DIR / RR_FILES[name]
    if not path.exists():
        raise _missing_file_error(path, name)
    df = pd.read_csv(path)
    df = df.drop(columns=df.columns[0])  # row index (first value is corrupted in Led.csv)
    target = df.columns[-1]
    y_raw = df[target].astype(str).str.strip().str.strip("'")
    y = _encode_target(y_raw, verbose_name=target)
    X_df = df.drop(columns=[target])
    for c in X_df.columns:  # numeric columns stored as text stay numeric
        if not pd.api.types.is_numeric_dtype(X_df[c]):
            conv = pd.to_numeric(X_df[c], errors="coerce")
            if conv.notna().mean() > 0.95:
                X_df[c] = conv
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]
    parts = [X_df[numeric_cols].apply(lambda s: s.fillna(s.median()))]
    if categorical_cols:
        parts.append(pd.get_dummies(X_df[categorical_cols].astype(str), drop_first=False))
    return pd.concat(parts, axis=1).to_numpy(dtype=float), y


def load_creditcard_ulb():
    """ULB / Worldline credit-card fraud data (Dal Pozzolo et al., 2015):
    284,807 European card transactions from September 2013, 492 frauds
    (0.172%), PCA components V1-V28 plus Time and Amount. Target: Class.
    """
    path = RAW_DIR / "credit_card" / "creditcard.csv"
    if not path.exists():
        raise _missing_file_error(path, "creditcard_ulb")
    df = pd.read_csv(path)
    y = df["Class"].astype(int).to_numpy()
    X = df.drop(columns=["Class"]).to_numpy(dtype=float)
    return X, y


REAL_DATASETS = {
    "communities_crime": load_communities_crime,
    "credit_card_fraud": load_credit_card_fraud,
    "synthetic_financial_fraud": load_synthetic_financial_fraud,
    "fraud_detection_transactions": load_fraud_detection_transactions,
    "credit_card_fraud_2023": load_credit_card_fraud_2023,
    "ailerons": load_ailerons,
    "segment0": load_segment0,
    "shuttle_2_vs_5": load_shuttle_2_vs_5,
    "default_credit_card_clients": load_default_credit_card_clients,
    "mammographic": load_mammographic,
    "mushrooms": load_mushrooms,
    "bank_marketing": load_bank_marketing,
    # bioassay, nih_chest_xray_sample: excluded per project decision.
    "creditcard_ulb": load_creditcard_ulb,
    **{k: (lambda k=k: load_relabeling_raking(k)) for k in RR_FILES},
}

# Datasets that guided development of v1-v3 (incl. the v3 design grid) vs.
# datasets first touched only after the v3 configuration was frozen.
DEVELOPMENT_DATASETS = [
    "communities_crime", "credit_card_fraud", "synthetic_financial_fraud", "fraud_detection_transactions",
    "credit_card_fraud_2023", "ailerons", "segment0", "shuttle_2_vs_5",
    "default_credit_card_clients", "mammographic", "mushrooms", "bank_marketing",
]
HOLDOUT_DATASETS = ["creditcard_ulb", *RR_FILES]
