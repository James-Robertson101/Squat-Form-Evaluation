"""
cross_validate.py
5-fold cross-validation for the best models:
  - Random Forest on the front view
  - Random Forest on the side view

Labels are remapped within each fold (train-only mapping applied to test).
Feature columns are declared explicitly — a stale CSV raises a clear error.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

#Paths
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


def remap_labels_fold(y_train: pd.DataFrame, y_test: pd.DataFrame):
    """
    Build label mapping from train fold only, apply to both.
    Test samples with classes unseen in train get NaN — they are skipped
    in the F1 calculation below.
    """
    y_train_r = y_train.copy()
    y_test_r  = y_test.copy()
    for col in y_train.columns:
        unique_vals = sorted(y_train[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        y_train_r[col] = y_train[col].map(mapping)
        y_test_r[col]  = y_test[col].map(mapping)   # unknown → NaN
    return y_train_r, y_test_r


# Cross-validation
def cross_validate_best(
    csv_path:     str,
    feature_cols: list,
    label_cols:   list,
    model,
    view_name:    str,
    model_name:   str,
    n_splits:     int = 5,
) -> None:

    print(f"  {view_name} — {model_name}  ({n_splits}-Fold CV): ")
    df = pd.read_csv(csv_path)
    validate_columns(df, feature_cols + label_cols, csv_path)
    X = df[feature_cols].copy()
    y = df[label_cols].copy()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    exact_scores      = []
    f1_scores         = []
    per_label_f1_all  = {col: [] for col in label_cols}
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
        X_tr = X.iloc[train_idx].reset_index(drop=True)
        X_te = X.iloc[test_idx].reset_index(drop=True)
        y_tr = y.iloc[train_idx].reset_index(drop=True)
        y_te = y.iloc[test_idx].reset_index(drop=True)
        y_tr_r, y_te_r = remap_labels_fold(y_tr, y_te)
        imputer = SimpleImputer(strategy='median')
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_te_sc = scaler.transform(imputer.transform(X_te))
        model.fit(X_tr_sc, y_tr_r)
        y_pred    = model.predict(X_te_sc)
        y_pred_df = pd.DataFrame(y_pred, columns=label_cols)
        exact = (y_pred_df == y_te_r).all(axis=1).mean()
        exact_scores.append(exact)

        fold_f1s = []
        for col in label_cols:
            mask = y_te_r[col].notna()
            if mask.sum() == 0:
                continue
            score = f1_score(
                y_te_r[col][mask],
                y_pred_df[col][mask],
                average='weighted',
                zero_division=0,
            )
            fold_f1s.append(score)
            per_label_f1_all[col].append(score)
        mean_f1 = np.mean(fold_f1s) if fold_f1s else 0.0
        f1_scores.append(mean_f1)
        print(f"  Fold {fold}: Exact={exact:.3f}  MeanF1={mean_f1:.3f}")

    print(f"\n  Summary")
    print(f"  Exact Match : {np.mean(exact_scores):.3f} ± {np.std(exact_scores):.3f}")
    print(f"  Mean F1     : {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print(f"\n  Per-label F1 (mean across folds):")
    for col in label_cols:
        scores = per_label_f1_all[col]
        if scores:
            print(f"    {col:30s}  {np.mean(scores):.3f} ± {np.std(scores):.3f}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    cross_validate_best(
        csv_path     = FRONT_CSV,
        feature_cols = FRONT_FEATURES,
        label_cols   = FRONT_LABELS,
        model        = MultiOutputClassifier(
                           RandomForestClassifier(
                                n_estimators=200,
                                max_depth=None,
                                max_features='sqrt',
                                min_samples_split=5,
                                random_state=42
                           )
                       ),
        view_name    = "Front",
        model_name   = "Random Forest",
    )

    cross_validate_best(
        csv_path     = SIDE_CSV,
        feature_cols = SIDE_FEATURES,
        label_cols   = SIDE_LABELS,
        model        = MultiOutputClassifier(
                           RandomForestClassifier(
                            n_estimators=200,
                            max_depth=None,
                            max_features='sqrt',
                            min_samples_split=2,
                            random_state=42
    )
                       ),
        view_name    = "Side",
        model_name   = "Random Forest",
    )
