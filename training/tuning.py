import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

warnings.filterwarnings('ignore')

# Load data
front = pd.read_csv(r"C:\Users\james\Squat Form Evaluation\datasets\front\front_view_merged.csv")
side  = pd.read_csv(r"C:\Users\james\Squat Form Evaluation\datasets\side\side_view_merged.csv")

front_labels = ["knee_valgus", "knee_varus", "lateral_hip_shift", "torso_lateral_lean", "foot_stability"]
side_labels  = ["squat_depth", "lumbar_flexion", "forward_lean", "descent_control", "ascent_sticking", "foot_stability"]

X_front = front.drop(columns=front_labels + ['video_name'], errors='ignore')
y_front = front[front_labels]

X_side = side.drop(columns=side_labels + ['video_name'], errors='ignore')
y_side = side[side_labels]

# Preprocessing
def preprocess(X):
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    return scaler.fit_transform(X_imputed)

X_front_p = preprocess(X_front)
X_side_p  = preprocess(X_side)


# Label remapping
class RemappingMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = pd.DataFrame(y) if not isinstance(y, pd.DataFrame) else y
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
            est_params = self.estimator.get_params(deep=True)
            for k, v in est_params.items():
                params[f'estimator__{k}'] = v
        return params

    def set_params(self, **params):
        estimator_params = {}
        for k, v in params.items():
            if k == 'estimator':
                self.estimator = v
            elif k.startswith('estimator__'):
                estimator_params[k[len('estimator__'):]] = v

        if estimator_params:
            self.estimator.set_params(**estimator_params)

        return self
# Score metric
def multioutput_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    scores = []
    for i in range(y_true.shape[1]):
        score = f1_score(
            y_true[:, i],
            y_pred[:, i],
            average='weighted',
            zero_division=0
        )
        scores.append(score)

    return np.mean(scores)

custom_scorer = make_scorer(multioutput_f1)
# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Parameter grid
rf_params = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [None, 10, 20],
    'estimator__min_samples_split': [2, 5, 10],
    'estimator__max_features': ['sqrt', 'log2']
}

# Tune front model
print("\nTuning Random Forest (Front)...")

rf_front = GridSearchCV(
    RemappingMultiOutputClassifier(
        RandomForestClassifier(random_state=42)
    ),
    param_grid=rf_params,
    cv=cv,
    scoring=custom_scorer,
    n_jobs=-1,
    verbose=1
)

rf_front.fit(X_front_p, y_front.values)

print(f"Front Best Params: {rf_front.best_params_}")
print(f"Front Best CV F1: {rf_front.best_score_:.3f}")

# Tune side model
print("\nTuning Random Forest (Side)...")

rf_side = GridSearchCV(
    RemappingMultiOutputClassifier(
        RandomForestClassifier(random_state=42)
    ),
    param_grid=rf_params,
    cv=cv,
    scoring=custom_scorer,
    n_jobs=-1,
    verbose=1
)

rf_side.fit(X_side_p, y_side.values)

print(f"Side Best Params: {rf_side.best_params_}")
print(f"Side Best CV F1: {rf_side.best_score_:.3f}")