"""Registry of the 10 baseline imbalance-handling methods plus Cross-Balance.

Every entry is an unfitted, sklearn-``clone``-able estimator exposing
``fit``/``predict``/``predict_proba``, so the benchmark harness can treat
them uniformly. Resampling methods are wired up via ``imblearn.pipeline.
Pipeline`` so resampling only ever happens on the training fold.
"""

from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression

from cross_balance import CrossBalanceClassifier

RANDOM_STATE = 42


def _lr():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def build_methods(random_state=RANDOM_STATE):
    """Return {name: unfitted estimator} for the fixed 10-method baseline
    set plus Cross-Balance (11 entries total).
    """
    return {
        "Baseline": LogisticRegression(max_iter=1000, random_state=random_state),
        "ClassWeighted": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        "RandomUnderSampler": ImbPipeline([
            ("resample", RandomUnderSampler(random_state=random_state)),
            ("clf", _lr()),
        ]),
        "RandomOverSampler": ImbPipeline([
            ("resample", RandomOverSampler(random_state=random_state)),
            ("clf", _lr()),
        ]),
        "SMOTE": ImbPipeline([
            ("resample", SMOTE(random_state=random_state)),
            ("clf", _lr()),
        ]),
        "BorderlineSMOTE": ImbPipeline([
            ("resample", BorderlineSMOTE(random_state=random_state)),
            ("clf", _lr()),
        ]),
        "ADASYN": ImbPipeline([
            ("resample", ADASYN(random_state=random_state)),
            ("clf", _lr()),
        ]),
        "SMOTETomek": ImbPipeline([
            ("resample", SMOTETomek(random_state=random_state)),
            ("clf", _lr()),
        ]),
        "BalancedRandomForest": BalancedRandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ),
        "EasyEnsemble": EasyEnsembleClassifier(n_estimators=10, random_state=random_state, n_jobs=-1),
        "CrossBalance": CrossBalanceClassifier(
            base_estimator="auto",
            ensemble_type="boosting", ratio=2.0,
            n_chunks=25, weighting="oof", random_state=random_state,
        ),
    }
