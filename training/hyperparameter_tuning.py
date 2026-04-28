"""
hyperparameter_tuning.py
Runs a group-aware grid search across six classifiers for both the front
and side view datasets. Groups by video name so no video leaks across
the train/test boundary.
"""

import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from xgboost import XGBClassifier
from Remapper import RemappingMultiOutputClassifier
warnings.filterwarnings('ignore')
 
FRONT_CSV = r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv"
SIDE_CSV  = r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv"

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
 
def validate_columns(df: pd.DataFrame, required: list, csv_path: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n[Column mismatch] in {csv_path}\n"
            f"  Missing: {missing}\n"
            f"  Re-run feature extraction to regenerate the CSV."
        )


def remap_labels(y_train: pd.DataFrame, y_test: pd.DataFrame):
    """
    Fits label encoders on training data only then applies to both splits.
    Unseen test labels get mapped to -1 so they fail visibly rather than silently.
    """
    y_train_r = y_train.copy()
    y_test_r  = y_test.copy()
    for col in y_train.columns:
        unique_vals = sorted(y_train[col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        y_train_r[col] = y_train[col].map(mapping)
        # unseen labels in test → map to -1 so they don't crash but are counted wrong
        y_test_r[col]  = y_test[col].map(mapping).fillna(-1).astype(int)
    return y_train_r, y_test_r


def multioutput_f1(y_true, y_pred):
    """mean weighted F1 across all output labels — used as the grid search scorer"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = [
        f1_score(y_true[:, i], y_pred[:, i],
                 average='weighted', zero_division=0)
        for i in range(y_true.shape[1])
    ]
    return float(np.mean(scores))


def exact_match(y_true, y_pred):
    return float((np.array(y_true) == np.array(y_pred)).all(axis=1).mean())


def hamming(y_true, y_pred):
    return float((np.array(y_true) != np.array(y_pred)).mean())


def make_pipeline(classifier):
    """
    Wraps imputer → scaler → classifier in a single Pipeline so preprocessing
    is always fitted on training data only, even inside CV folds.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     RemappingMultiOutputClassifier(classifier)),
    ])

# param_grid keys are prefixed with clf__estimator__ because they pass
# through Pipeline → RemappingMultiOutputClassifier → the actual estimator
MODEL_CONFIGS = [
    (
        "Random Forest",
        RandomForestClassifier(random_state=42, class_weight='balanced'),
        {
            'clf__estimator__n_estimators':     [100, 200, 300],
            'clf__estimator__max_depth':        [None, 5, 10, 15],
            'clf__estimator__max_features':     ['sqrt', 'log2'],
            'clf__estimator__min_samples_split':[2, 5],
            'clf__estimator__min_samples_leaf': [1, 2],
        },
    ),
    (
        "XGBoost",
        XGBClassifier(eval_metric='mlogloss', random_state=42,
                      use_label_encoder=False),
        {
            'clf__estimator__n_estimators':  [100, 200, 300],
            'clf__estimator__max_depth':     [3, 5, 7],
            'clf__estimator__learning_rate': [0.05, 0.1, 0.2],
            'clf__estimator__subsample':     [0.8, 1.0],
        },
    ),
    (
        "SVM",
        SVC(kernel='rbf', probability=True, random_state=42,
            class_weight='balanced'),
        {
            'clf__estimator__C':     [0.1, 1, 10, 100],
            'clf__estimator__gamma': ['scale', 'auto'],
        },
    ),
    (
        "KNN",
        KNeighborsClassifier(),
        {
            'clf__estimator__n_neighbors': [3, 5, 7, 11],
            'clf__estimator__weights':     ['uniform', 'distance'],
            'clf__estimator__metric':      ['euclidean', 'manhattan'],
        },
    ),
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42,
                           class_weight='balanced'),
        {
            'clf__estimator__C':       [0.01, 0.1, 1, 10],
            'clf__estimator__solver':  ['lbfgs', 'saga'],
            'clf__estimator__penalty': ['l2'],
        },
    ),
    (
        "MLP",
        MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
        {
            'clf__estimator__hidden_layer_sizes': [(64, 32), (128, 64), (64, 64, 32)],
            'clf__estimator__alpha':              [0.0001, 0.001, 0.01],
            'clf__estimator__learning_rate_init': [0.001, 0.01],
        },
    ),
]

def run_search(csv_path: str, feature_cols: list, label_cols: list,
               view_name: str):

    print("\n")
    print(f"  {view_name} View — Group-aware hyperparameter search")
    df = pd.read_csv(csv_path)
    validate_columns(df, feature_cols + label_cols + ['video_name'], csv_path)

    groups = df['video_name'].values
    X      = df[feature_cols]
    y      = df[label_cols]

    # group-aware split — no video appears in both train and test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test   = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test   = y.iloc[train_idx], y.iloc[test_idx]
    groups_train      = groups[train_idx]

    y_train_r, y_test_r = remap_labels(y_train, y_test)

    print(f"  Train samples: {len(X_train)}  |  Test samples: {len(X_test)}")
    print(f"  Unique train videos: {len(np.unique(groups_train))}  "
          f"|  Unique test videos: {len(np.unique(groups[test_idx]))}")

    inner_cv = GroupKFold(n_splits=5)

    from sklearn.metrics import make_scorer
    scorer = make_scorer(multioutput_f1)

    best_results = []
    for name, base_est, param_grid in MODEL_CONFIGS:
        print(f"\n  Searching {name}...")
        pipeline = make_pipeline(base_est)

        try:
            gs = GridSearchCV(
                pipeline,
                param_grid  = param_grid,
                cv          = inner_cv,
                scoring     = scorer,
                n_jobs      = -1,
                verbose     = 0,
                refit       = True,
                error_score = 0,
            )
            gs.fit(X_train, y_train_r, groups=groups_train)

            best_cv_f1  = gs.best_score_
            best_params = gs.best_params_
            best_pipe   = gs.best_estimator_
            y_pred = best_pipe.predict(X_test)
            test_f1  = multioutput_f1(y_test_r.values, y_pred)
            test_acc = exact_match(y_test_r.values, y_pred)
            test_ham = hamming(y_test_r.values, y_pred)
            y_pred_df    = pd.DataFrame(y_pred, columns=label_cols)
            y_test_reset = y_test_r.reset_index(drop=True)
            per_label_f1 = {
                col: f1_score(y_test_reset[col], y_pred_df[col],
                              average='weighted', zero_division=0)
                for col in label_cols
            }

            best_results.append({
                'name':      name,
                'cv_f1':     best_cv_f1,
                'test_f1':   test_f1,
                'test_acc':  test_acc,
                'test_ham':  test_ham,
                'params':    best_params,
                'per_label': per_label_f1,
                'pipeline':  best_pipe,
            })

            print(f"    Best CV F1   : {best_cv_f1:.3f}")
            print(f"    Test F1      : {test_f1:.3f}")
            print(f"    Exact Match  : {test_acc:.3f}")
            print(f"    Hamming Loss : {test_ham:.3f}")
            print(f"    Best params  : {best_params}")
            print(f"    Per-label F1 :")
            for col, score in per_label_f1.items():
                print(f"      {col:30s}  {score:.3f}")

        except Exception as exc:
            print(f"    FAILED: {exc}")
            best_results.append({
                'name': name, 'cv_f1': 0, 'test_f1': 0,
                'test_acc': 0, 'test_ham': 1,
                'params': {}, 'per_label': {}, 'pipeline': None,
            })

    # majority vote across the top 3 by CV F1
    print(f"\n  Building ensemble from top-3 CV performers...")
    top3 = sorted(best_results, key=lambda r: r['cv_f1'], reverse=True)[:3]
    print(f"  Ensemble members: {[r['name'] for r in top3]}")

    try:
        # GridSearchCV already refits on full train so pipelines are ready to use
        ensemble_preds = np.array([
            r['pipeline'].predict(X_test) for r in top3
        ])  # shape (3, n_test, n_labels)

        # Majority vote per label per sample
        from scipy.stats import mode
        voted = np.apply_along_axis(
            lambda x: mode(x, keepdims=True).mode[0], 0, ensemble_preds
        )  # shape (n_test, n_labels)

        ens_f1  = multioutput_f1(y_test_r.values, voted)
        ens_acc = exact_match(y_test_r.values, voted)
        ens_ham = hamming(y_test_r.values, voted)

        print(f"    Ensemble Test F1      : {ens_f1:.3f}")
        print(f"    Ensemble Exact Match  : {ens_acc:.3f}")
        print(f"    Ensemble Hamming Loss : {ens_ham:.3f}")

        best_results.append({
            'name': f"Ensemble ({'+'.join(r['name'] for r in top3)})",
            'cv_f1': np.mean([r['cv_f1'] for r in top3]),
            'test_f1': ens_f1,
            'test_acc': ens_acc,
            'test_ham': ens_ham,
            'params': {}, 'per_label': {}, 'pipeline': None,
        })

    except Exception as exc:
        print(f"    Ensemble FAILED: {exc}")

    # Summary table 
    print("\n")
    print(f"  {view_name} — Final Rankings (by Test F1)")
    print(f"  {'Model':<40} {'CV F1':>6}  {'Test F1':>7}  "
          f"{'Exact':>5}  {'Hamming':>7}")
    print(f"  {'─'*40} {'─'*6}  {'─'*7}  {'─'*5}  {'─'*7}")
    for r in sorted(best_results, key=lambda x: x['test_f1'], reverse=True):
        print(f"  {r['name']:<40} {r['cv_f1']:>6.3f}  {r['test_f1']:>7.3f}  "
              f"{r['test_acc']:>5.3f}  {r['test_ham']:>7.3f}")

    best = max(best_results, key=lambda r: r['test_f1'])
    print(f"\n  ✓ Best model for {view_name}: {best['name']}")
    print(f"    Test F1={best['test_f1']:.3f}  "
          f"Exact={best['test_acc']:.3f}  "
          f"Hamming={best['test_ham']:.3f}")
    if best['params']:
        print(f"    Params: {best['params']}")

    return best_results


# Entry point
if __name__ == "__main__":
    front_results = run_search(FRONT_CSV, FRONT_FEATURES, FRONT_LABELS, "Front")
    side_results  = run_search(SIDE_CSV,  SIDE_FEATURES,  SIDE_LABELS,  "Side")

    print("\n\n" + "="*65)
    print("  FINAL SUMMARY")
    print("="*65)

    for view, results in [("Front", front_results), ("Side", side_results)]:
        best = max(results, key=lambda r: r['test_f1'])
        print(f"\n  {view} View  →  Best: {best['name']}")
        print(f"    CV F1={best['cv_f1']:.3f}  Test F1={best['test_f1']:.3f}  "
              f"Exact={best['test_acc']:.3f}  Hamming={best['test_ham']:.3f}")