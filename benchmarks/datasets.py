"""Dataset loaders for the benchmark harness.

Phase A: synthetic imbalanced data only. Phase B will add loaders for the
real-world datasets (Communities & Crime, Credit Card Fraud, etc.) reading
from ``data/raw/<dataset_name>/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Named imbalance ratios the project benchmarks across (minority fraction).
SYNTHETIC_RATIOS = {
    "99-1": 0.01,
    "99-2": 0.02,
    "99-5": 0.05,
}


def make_synthetic_imbalanced(
    ratio=0.01,
    n_samples=20_000,
    n_features=20,
    n_informative=18,
    n_redundant=2,
    n_clusters_per_class=3,
    label_noise=0.005,
    random_state=42,
):
    """Generate a synthetic binary-classification dataset with a fixed
    minority-class fraction.

    Mirrors the configuration validated in the v1 prototype
    (``cbc_v2.py``: 20 features, 18 informative, clustered classes, small
    label-noise injection), generalized to any minority ``ratio``.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        weights=[1 - ratio, ratio],
        flip_y=0,  # noise injected explicitly below, not via sklearn's own flip
        random_state=random_state,
    )

    if label_noise > 0:
        rng = np.random.RandomState(random_state)
        majority_idx = np.where(y == 0)[0]
        n_flip = int(len(majority_idx) * label_noise)
        flip_idx = rng.choice(majority_idx, size=n_flip, replace=False)
        y = y.copy()
        y[flip_idx] = 1

    return X, y


def load_synthetic(name, use_cache=True, **kwargs):
    """Load (and cache) a named synthetic dataset, e.g. ``load_synthetic('99-1')``."""
    if name not in SYNTHETIC_RATIOS:
        raise ValueError(f"Unknown synthetic dataset '{name}'. Choose from {list(SYNTHETIC_RATIOS)}")

    n_samples = kwargs.get("n_samples", 20_000)
    random_state = kwargs.get("random_state", 42)
    cache_path = PROCESSED_DIR / "synthetic" / f"{name}_n{n_samples}_rs{random_state}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        X = df.drop(columns=["target"]).to_numpy()
        y = df["target"].to_numpy()
        return X, y

    X, y = make_synthetic_imbalanced(ratio=SYNTHETIC_RATIOS[name], **kwargs)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y
        df.to_parquet(cache_path, index=False)

    return X, y
