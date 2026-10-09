"""Gradient-boosted pairwise matcher (LightGBM, MIT license)."""
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 255,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
    "n_jobs": -1,
}


class MatchModel:
    def __init__(self, params: Optional[dict] = None, num_boost_round: int = 4000, early_stopping: int = 150):
        self.params = dict(LGB_PARAMS, **(params or {}))
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.feature_names: List[str] = []
        self.booster = None
        self.sk_model = None
        self.best_iteration = 0

    def fit(self, X_tr: pd.DataFrame, y_tr, X_va: pd.DataFrame, y_va, sample_weight: Optional[np.ndarray] = None) -> "MatchModel":
        self.feature_names = list(X_tr.columns)
        y_tr = np.asarray(y_tr, dtype=np.float32)
        y_va = np.asarray(y_va, dtype=np.float32)
        if HAS_LGB:
            dtr = lgb.Dataset(X_tr.to_numpy(np.float32), label=y_tr, weight=sample_weight,
                              feature_name=self.feature_names, free_raw_data=True)
            dva = lgb.Dataset(X_va.to_numpy(np.float32), label=y_va, reference=dtr, free_raw_data=True)
            self.booster = lgb.train(
                self.params, dtr, num_boost_round=self.num_boost_round, valid_sets=[dva], valid_names=["val"],
                callbacks=[lgb.early_stopping(self.early_stopping, verbose=False), lgb.log_evaluation(250)],
            )
            self.best_iteration = self.booster.best_iteration or self.booster.current_iteration()
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.sk_model = HistGradientBoostingClassifier(
                max_iter=1500, learning_rate=0.05, max_leaf_nodes=63, early_stopping=True,
                validation_fraction=0.1, random_state=42)
            self.sk_model.fit(X_tr.to_numpy(np.float32), y_tr, sample_weight=sample_weight)
            self.best_iteration = self.sk_model.n_iter_
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0, dtype=np.float64)
        arr = X[self.feature_names].to_numpy(np.float32)
        if self.booster is not None:
            return self.booster.predict(arr, num_iteration=self.best_iteration)
        return self.sk_model.predict_proba(arr)[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        if self.booster is not None:
            imp = self.booster.feature_importance(importance_type="gain")
        else:
            imp = np.zeros(len(self.feature_names))
        return (pd.DataFrame({"feature": self.feature_names, "gain": imp})
                .sort_values("gain", ascending=False).reset_index(drop=True))

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "MatchModel":
        with open(path, "rb") as f:
            return pickle.load(f)
