"""
save_models.py
Trains and saves the final front-view (Random Forest) and side-view (RandomForest)
models together with their fitted imputer and scaler.

Feature columns are declared explicitly so a stale CSV raises a clear error
instead of silently training on the wrong columns.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Paths
FRONT_CSV   = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
SIDE_CSV    = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"
MODELS_DIR  = "models"

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

# Label columns
FRONT_LABELS = ["knee_valgus", "knee_varus", "lateral_hip_shift",
                "torso_lateral_lean", "foot_stability"]

SIDE_LABELS  = ["squat_depth", "lumbar_flexion", "forward_lean",
                "descent_control", "ascent_sticking", "foot_stability"]


# Helpers
def validate_columns(df: pd.DataFrame, required: list, csv_path: str) -> None:
    """Raise a clear error if any expected feature column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[Column mismatch] in {csv_path}\n"
            f"  Missing columns: {missing}\n"
            f"  This usually means the CSV was built with an older extractor.\n"
            f"  Re-run feature extraction to regenerate the CSV."
        )


def remap_labels(y: pd.DataFrame):
    """
    Map each label column to consecutive 0-based integers.
    Returns the remapped DataFrame and the encoder dict for later decoding.
    """
    y_remapped = y.copy()
    encoders = {}
    for col in y.columns:
        unique_vals = sorted(y[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        y_remapped[col] = y[col].map(mapping)
        encoders[col] = mapping
    return y_remapped, encoders


def fit_and_save(
    csv_path:    str,
    feature_cols: list,
    label_cols:   list,
    model,
    prefix:       str,
    out_dir:      str,
) -> None:
    """
    Load CSV → validate columns → fit imputer/scaler/model → save artefacts.
    Saves four files: _model.pkl, _imputer.pkl, _scaler.pkl, _encoders.pkl
    Also saves the ordered feature column list as _feature_cols.pkl so
    the inference pipeline can reindex incoming data without guessing.
    """
    print(f"  Training {prefix} model  ({csv_path}): ")
    df = pd.read_csv(csv_path)
    validate_columns(df, feature_cols + label_cols, csv_path)
    X = df[feature_cols].copy()
    y = df[label_cols].copy()
    print(f"  Samples : {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Labels  : {label_cols}")
    y_remapped, encoders = remap_labels(y)
    print(f"  Label encoders: { {k: list(v.keys()) for k, v in encoders.items()} }")
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(imputer.fit_transform(X))
    model.fit(X_sc, y_remapped)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model,        os.path.join(out_dir, f"{prefix}_model.pkl"))
    joblib.dump(imputer,      os.path.join(out_dir, f"{prefix}_imputer.pkl"))
    joblib.dump(scaler,       os.path.join(out_dir, f"{prefix}_scaler.pkl"))
    joblib.dump(encoders,     os.path.join(out_dir, f"{prefix}_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(out_dir, f"{prefix}_feature_cols.pkl"))
    print(f"  Saved → {out_dir}/{prefix}_*.pkl")

# Main 
if __name__ == "__main__":

    fit_and_save(
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
        prefix       = "front",
        out_dir      = MODELS_DIR,
    )

    fit_and_save(
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
        prefix       = "side",
        out_dir      = MODELS_DIR,
    )
    print("  All models saved.  Contents of models/:")
    for f in sorted(os.listdir(MODELS_DIR)):
        print(f"    {f}")