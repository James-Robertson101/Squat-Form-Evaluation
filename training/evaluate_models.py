"""
evaluate_models.py
Benchmarks six classifiers on the front and side datasets using a single
80/20 train-test split.  Feature columns are declared explicitly so a stale
CSV raises a clear error instead of silently training on the wrong data.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
FRONT_CSV = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
SIDE_CSV  = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"

# ── Feature columns (must match extract_squat_features_from_frames output) ─────
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

#Models
MODELS = {
    "Random Forest":       MultiOutputClassifier(RandomForestClassifier(n_estimators=200, random_state=42)),
    "XGBoost":             MultiOutputClassifier(XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42)),
    "SVM":                 MultiOutputClassifier(SVC(kernel='rbf', probability=True, random_state=42)),
    "KNN":                 MultiOutputClassifier(KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42)),
    "MLP":                 MultiOutputClassifier(MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
}


#Helpers 
def validate_columns(df: pd.DataFrame, required: list, csv_path: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[Column mismatch] in {csv_path}\n"
            f"  Missing: {missing}\n"
            f"  Re-run feature extraction to regenerate the CSV."
        )


def remap_labels(y: pd.DataFrame):
    y_remapped = y.copy()
    encoders = {}
    for col in y.columns:
        unique_vals = sorted(y[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        y_remapped[col] = y[col].map(mapping)
        encoders[col] = mapping
    return y_remapped, encoders

def print_feature_importance(model, feature_names, model_name, view_name):
    """
    Prints top feature importances for tree-based models.
    """
    try:
        # MultiOutputClassifier wrapper → access estimator inside
        estimator = model.estimators_[0]

        if hasattr(estimator, "feature_importances_"):
            importances = np.mean(
                [est.feature_importances_ for est in model.estimators_],
                axis=0
            )

            importance_df = pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            }).sort_values("importance", ascending=False)

            print(f"\n  🔍 Feature Importance — {model_name} ({view_name})")
            for _, row in importance_df.iterrows():
                print(f"    {row['feature']:30s} {row['importance']:.4f}")

    except Exception as e:
        print(f"\n  Could not compute feature importance for {model_name}: {e}")

# Evaluation
def evaluate_models(
    csv_path:     str,
    feature_cols: list,
    label_cols:   list,
    view_name:    str,
) -> dict:
    print(f"  {view_name} View  —  80/20 train-test split")
    df = pd.read_csv(csv_path)
    validate_columns(df, feature_cols + label_cols, csv_path)

    X = df[feature_cols].copy()
    y = df[label_cols].copy()

    y_r, _ = remap_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_r, test_size=0.2, random_state=42
    )

    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_sc  = scaler.transform(imputer.transform(X_test))

    results = {}

    for name, model in MODELS.items():
        try:
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            if name in ["Random Forest", "XGBoost"]:
                print_feature_importance(model, feature_cols, name, view_name)

            y_pred_df    = pd.DataFrame(y_pred,  columns=label_cols)
            y_test_reset = y_test.reset_index(drop=True)

            exact_acc     = (y_pred_df == y_test_reset).all(axis=1).mean()
            hamming       = (y_pred_df != y_test_reset).values.mean()
            per_label_acc = (y_pred_df == y_test_reset).mean()

            per_label_f1 = {
                col: f1_score(y_test_reset[col], y_pred_df[col],
                              average='weighted', zero_division=0)
                for col in label_cols
            }
            mean_f1 = np.mean(list(per_label_f1.values()))

            results[name] = {
                "Exact Match Acc": exact_acc,
                "Hamming Loss":    hamming,
                "Mean F1":         mean_f1,
            }

            print(f"\n  {name}")
            print(f"    Exact Match Accuracy : {exact_acc:.3f}")
            print(f"    Hamming Loss         : {hamming:.3f}")
            print(f"    Mean Weighted F1     : {mean_f1:.3f}")
            print(f"    Per-label breakdown  :")
            for col in label_cols:
                print(f"      {col:30s}  acc={per_label_acc[col]:.3f}  "
                      f"f1={per_label_f1[col]:.3f}")

        except Exception as exc:
            print(f"\n  {name} — FAILED: {exc}")
            results[name] = {"Exact Match Acc": 0, "Hamming Loss": 1, "Mean F1": 0}

    # Rankings
    print(f"\n--- {view_name} Ranking (by Exact Match Accuracy) ---")
    for rank, (name, sc) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["Exact Match Acc"], reverse=True), 1
    ):
        print(f"  {rank}. {name:25s}  Exact={sc['Exact Match Acc']:.3f}  "
              f"Hamming={sc['Hamming Loss']:.3f}  MeanF1={sc['Mean F1']:.3f}")

    print(f"\n--- {view_name} Ranking (by Mean Weighted F1) ---")
    for rank, (name, sc) in enumerate(
        sorted(results.items(), key=lambda x: x[1]["Mean F1"], reverse=True), 1
    ):
        print(f"  {rank}. {name:25s}  MeanF1={sc['Mean F1']:.3f}  "
              f"Exact={sc['Exact Match Acc']:.3f}  Hamming={sc['Hamming Loss']:.3f}")

    return results


# Main
if __name__ == "__main__":
    evaluate_models(FRONT_CSV, FRONT_FEATURES, FRONT_LABELS, "Front")
    evaluate_models(SIDE_CSV,  SIDE_FEATURES,  SIDE_LABELS,  "Side")
