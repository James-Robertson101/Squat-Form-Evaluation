import pandas as pd
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.base import BaseEstimator, ClassifierMixin, clone
#remapping helper function used in both save_models.py and hyperparamter_tuning.py
#remapping is done as some models e.g. XGBOOST require 0,1,2 for classification
class RemappingMultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps a single base estimator inside MultiOutputClassifier and remaps
    labels to consecutive 0-based integers on every fit() call.
    This prevents XGBoost / other classifiers from crashing on non-consecutive
    class indices that appear when a CV fold happens to be missing a class.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y = pd.DataFrame(y).reset_index(drop=True)
        self.encoders_ = {}
        y_r = y.copy()
        for col in y.columns:
            uniq = sorted(y[col].unique())
            mapping = {v: i for i, v in enumerate(uniq)}
            self.encoders_[col] = mapping
            y_r[col] = y[col].map(mapping)
        self.model_ = MultiOutputClassifier(clone(self.estimator))
        self.model_.fit(X, y_r)
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
