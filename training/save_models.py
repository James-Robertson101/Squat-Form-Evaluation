"""
save_models.py

Trains and saves the final models using the best hyperparameters found
during group-aware hyperparameter search:

  Front view  →  KNN   (Test F1=0.817, Exact=0.437, Hamming=0.166)
  Side view   →  Random Forest  (Test F1=0.752, Exact=0.319, Hamming=0.227)

The full sklearn Pipeline (imputer → scaler → classifier) is saved as a
single .pkl file per view, so inference just calls pipeline.predict(X)
without needing to manage separate preprocessing steps.

Also saves:
  - _encoders.pkl       : label mappings for decoding predictions
  - _feature_cols.pkl   : ordered feature list for reindexing inference data
  - _label_cols.pkl     : ordered label list for reconstructing output DataFrame
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from Remapper import RemappingMultiOutputClassifier

# ── Paths ─────────────────────────────────────────────────────────────────────
FRONT_CSV  = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
SIDE_CSV   = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"
MODELS_DIR = "models"

# ── Feature columns ───────────────────────────────────────────────────────────
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

# ── Label columns ─────────────────────────────────────────────────────────────
FRONT_LABELS = ["knee_valgus", "knee_varus", "lateral_hip_shift",
                "torso_lateral_lean", "foot_stability"]

SIDE_LABELS  = ["squat_depth", "lumbar_flexion", "forward_lean",
                "descent_control", "ascent_sticking", "foot_stability"]

# ── Best hyperparameters from group-aware search ──────────────────────────────
# Front: KNN  (Test F1=0.817, Exact Match=0.437, Hamming=0.166)
# FRONT_MODEL = MultiOutputClassifier(
#     KNeighborsClassifier(
#         n_neighbors=7,
#         weights='distance',
#         metric='manhattan',
#     )
# )
#defining models for ensemble classifier
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    eval_metric='mlogloss',
    random_state=42,
    use_label_encoder=False
)

knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='manhattan'
)

front_ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('knn', knn),
        ('xgb',xgb),
    ],
    voting='hard'  # majority vote (matches CV logic)
)
FRONT_MODEL = RemappingMultiOutputClassifier(front_ensemble)
# Side: Random Forest  (Test F1=0.752, Exact Match=0.319, Hamming=0.227)
SIDE_MODEL = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        max_features='sqrt',
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced_subsample',
    )
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def validate_columns(df: pd.DataFrame, required: list, csv_path: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[Column mismatch] in {csv_path}\n"
            f"  Missing columns: {missing}\n"
            f"  Re-run feature extraction to regenerate the CSV."
        )


def remap_labels(y: pd.DataFrame):
    """
    Map each label column to consecutive 0-based integers.
    Fit on the full dataset since we are saving the final production model.
    Returns remapped DataFrame and encoder dict {col: {original: encoded}}.
    """
    y_remapped = y.copy()
    encoders = {}
    for col in y.columns:
        unique_vals = sorted(y[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        y_remapped[col] = y[col].map(mapping)
        encoders[col] = mapping
    return y_remapped, encoders


# ── Build pipeline ────────────────────────────────────────────────────────────
def build_pipeline(classifier) -> Pipeline:
    """
    Returns imputer → scaler → classifier as a single Pipeline.
    Keeping preprocessing inside the pipeline means inference code
    only needs to call pipeline.predict(raw_feature_df).
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     classifier),
    ])


# ── Fit and save ──────────────────────────────────────────────────────────────
def fit_and_save(
    csv_path:     str,
    feature_cols: list,
    label_cols:   list,
    classifier,
    prefix:       str,
    out_dir:      str,
) -> None:

    model_name = type(classifier.estimator).__name__ if hasattr(classifier, 'estimator') \
                 else type(classifier).__name__

    print(f"\n  Training {prefix} model  [{model_name}]")
    print(f"  CSV : {csv_path}")

    df = pd.read_csv(csv_path)
    validate_columns(df, feature_cols + label_cols, csv_path)

    X = df[feature_cols].copy()
    y = df[label_cols].copy()

    print(f"  Samples  : {len(X)}")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Labels   : {label_cols}")

    y_remapped, encoders = remap_labels(y)
    print(f"  Encoders : { {k: list(v.keys()) for k, v in encoders.items()} }")

    # Build and fit full pipeline on ALL data (this is the production model)
    pipeline = build_pipeline(classifier)
    pipeline.fit(X, y_remapped)

    # Save artefacts
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipeline,     os.path.join(out_dir, f"{prefix}_pipeline.pkl"))
    joblib.dump(encoders,     os.path.join(out_dir, f"{prefix}_encoders.pkl"))
    joblib.dump(feature_cols, os.path.join(out_dir, f"{prefix}_feature_cols.pkl"))
    joblib.dump(label_cols,   os.path.join(out_dir, f"{prefix}_label_cols.pkl"))

    print(f"  Saved → {out_dir}/{prefix}_pipeline.pkl  (+ encoders, feature_cols, label_cols)")


# ── Inference helper (shows how to use saved models) ─────────────────────────
def predict_from_features(prefix: str, feature_dict: dict,
                           out_dir: str = MODELS_DIR) -> dict:
    """
    Example inference call.  Pass a dict of {feature_name: value} for one squat.
    Returns a dict of {label_name: predicted_class}.

    Usage:
        result = predict_from_features('front', {'valgus_min': 1.2, ...})
    """
    pipeline     = joblib.load(os.path.join(out_dir, f"{prefix}_pipeline.pkl"))
    encoders     = joblib.load(os.path.join(out_dir, f"{prefix}_encoders.pkl"))
    feature_cols = joblib.load(os.path.join(out_dir, f"{prefix}_feature_cols.pkl"))
    label_cols   = joblib.load(os.path.join(out_dir, f"{prefix}_label_cols.pkl"))

    X = pd.DataFrame([feature_dict])[feature_cols]
    y_pred = pipeline.predict(X)[0]

    # Decode back to original label values
    decoders = {col: {v: k for k, v in enc.items()} for col, enc in encoders.items()}
    return {col: decoders[col].get(pred, pred)
            for col, pred in zip(label_cols, y_pred)}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    fit_and_save(
        csv_path     = FRONT_CSV,
        feature_cols = FRONT_FEATURES,
        label_cols   = FRONT_LABELS,
        classifier   = FRONT_MODEL,
        prefix       = "front",
        out_dir      = MODELS_DIR,
    )

    fit_and_save(
        csv_path     = SIDE_CSV,
        feature_cols = SIDE_FEATURES,
        label_cols   = SIDE_LABELS,
        classifier   = SIDE_MODEL,
        prefix       = "side",
        out_dir      = MODELS_DIR,
    )

    print("\n  All models saved.  Contents of models/:")
    for f in sorted(os.listdir(MODELS_DIR)):
        size_kb = os.path.getsize(os.path.join(MODELS_DIR, f)) / 1024
        print(f"    {f:<40}  {size_kb:>8.1f} KB")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\n  Sanity check — predicting first row of each dataset:")

    for prefix, csv, features in [
        ("front", FRONT_CSV, FRONT_FEATURES),
        ("side",  SIDE_CSV,  SIDE_FEATURES),
    ]:
        df      = pd.read_csv(csv)
        row     = df[features].iloc[0].to_dict()
        result  = predict_from_features(prefix, row)
        print(f"\n  {prefix.capitalize()} prediction:")
        for label, value in result.items():
            print(f"    {label:<30s}  →  {value}")