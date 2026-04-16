"""
hyperparameter_tuning.py
Grid-searches the best hyperparameters for:
  - Random Forest  on the front view
  - XGBoost        on the side view

Uses a RemappingMultiOutputClassifier wrapper so label indices are always
consecutive integers within each CV fold, which XGBoost requires.
Feature columns are declared explicitly — a stale CSV will raise a clear error.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# Paths
FRONT_CSV = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
SIDE_CSV  = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"

# Feature columns
FRONT_FEATURES = [
    "valgus_min", "valgus_max", "valgus_variation",
    "torso_lateral_peak", "symmetry_mean",
    "heel_wobble", "heel_instability", "toe_instability",
    "knee_cave_frames", "knee_cave_frac",
    "knee_asym_mean", "knee_asym_std",
    "ankle_width_mean", "ankle_width_std",
    "hip_shift_max", "hip_shift_std",
    "sho_hip_offset_mean", "sho_hip_offset_max",
]

SIDE_FEATURES = [
    "hip_rom", "knee_rom", "torso_stability",
    "heel_instability", "toe_instability",
    "knee_min_angle", "hip_min_angle",
    "torso_lean_peak", "torso_lean_mean",
    "descent_frames", "ascent_frames", "descent_ascent_ratio",
    "knee_over_toe_mean", "knee_over_toe_max",
    "hip_below_knee_frac",
]

FRONT_LABELS = ["knee_valgus", "knee_varus", "lateral_hip_shift",
                "torso_lateral_lean", "foot_stability"]

SIDE_LABELS  = ["squat_depth", "lumbar_flexion", "forward_lean",
                "descent_control", "ascent_sticking", "foot_stability"]


# Helpers
def validate_columns(df: pd.DataFrame, required: list, csv_path: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[Column mismatch] in {csv_path}\n"
            f"  Missing: {missing}\n"
            f"  Re-run feature extraction to regenerate the CSV."
        )


def preprocess(X: pd.DataFrame):
    """Fit imputer + scaler on X and return transformed array."""
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    return scaler.fit_transform(imputer.fit_transform(X))


# Remapping wrapper (could in theory be taken out, was used for XGBOOST initially however I changed the feature extraction algorithm and now RandomForest is best for both)
class RemappingMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps MultiOutputClassifier and remaps label columns to consecutive
    0-based integers on every call to fit(), based only on the training data.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y.reset_index(drop=True)
        self.encoders_ = {}
        y_remapped = y.copy()
        for col in y.columns:
            unique_vals = sorted(y[col].unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            self.encoders_[col] = mapping
            y_remapped[col] = y[col].map(mapping)
        self.model_ = MultiOutputClassifier(self.estimator)
        self.model_.fit(X, y_remapped)
        self.columns_ = list(y.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def get_params(self, deep=True):
        params = {'estimator': self.estimator}
        if deep:
            for k, v in self.estimator.get_params(deep=True).items():
                params[f'estimator__{k}'] = v
        return params

    def set_params(self, **params):
        est_params = {}
        for k, v in params.items():
            if k == 'estimator':
                self.estimator = v
            elif k.startswith('estimator__'):
                est_params[k[len('estimator__'):]] = v
        if est_params:
            self.estimator.set_params(**est_params)
        return self


# Custom scorer: mean weighted F1 across all output labels
def multioutput_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = [
        f1_score(y_true[:, i], y_pred[:, i], average='weighted', zero_division=0)
        for i in range(y_true.shape[1])
    ]
    return np.mean(scores)

custom_scorer = make_scorer(multioutput_f1)
CV            = KFold(n_splits=5, shuffle=True, random_state=42)


# Tune Random Forest — Front view
def tune_front():
    print("  Tuning Random Forest — Front View: ")
    df = pd.read_csv(FRONT_CSV)
    validate_columns(df, FRONT_FEATURES + FRONT_LABELS, FRONT_CSV)

    X = preprocess(df[FRONT_FEATURES])
    y = df[FRONT_LABELS]

    # RF does not need the remapping wrapper because it accepts any integers,
    # but we use it anyway for consistency across both tuning runs.
    param_grid = {
        'estimator__n_estimators':    [100, 200, 300],
        'estimator__max_depth':       [None, 5, 10],
        'estimator__max_features':    ['sqrt', 'log2'],
        'estimator__min_samples_split': [2, 5],
    }

    grid = GridSearchCV(
        RemappingMultiOutputClassifier(RandomForestClassifier(random_state=42)),
        param_grid = param_grid,
        cv         = CV,
        scoring    = custom_scorer,
        n_jobs     = -1,
        verbose    = 1,
    )
    grid.fit(X, y)

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.3f}")
    return grid.best_params_, grid.best_score_


def tune_side():
    print("  Tuning Random Forest — Side View")
    df = pd.read_csv(SIDE_CSV)
    validate_columns(df, SIDE_FEATURES + SIDE_LABELS, SIDE_CSV)
    X = preprocess(df[SIDE_FEATURES])
    y = df[SIDE_LABELS]
    param_grid = {
        'estimator__n_estimators':    [100, 200, 300],
        'estimator__max_depth':       [None, 5, 10],
        'estimator__max_features':    ['sqrt', 'log2'],
        'estimator__min_samples_split': [2, 5],
    }
    grid = GridSearchCV(
        RemappingMultiOutputClassifier(
            RandomForestClassifier(random_state=42)
        ),
        param_grid=param_grid,
        cv=CV,
        scoring=custom_scorer,
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X, y)
    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.3f}")
    return grid.best_params_, grid.best_score_

if __name__ == "__main__":
    front_params, front_score = tune_front()
    side_params,  side_score  = tune_side()

    print("\n Summary of best hyperparameters: ")
    print(f"  Front (RF)     CV F1={front_score:.3f}  params={front_params}")
    print(f"  Side  (XGB)    CV F1={side_score:.3f}  params={side_params}")
