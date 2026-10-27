"""
High-Capacity Spatial-Spectral Ensemble Classifier Suite.
Provides tuned Random Forest, Extra Trees, and HistGradientBoosting models
capable of modeling non-linear physiological interactions across bands.
"""

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score


class HighCapacityEnsemble:
    """Tuned ensemble classifier optimized for spatial-spectral representations."""

    def __init__(self, model_type: str = "extra_trees", random_seed: int = 42):
        self.model_type = model_type.lower()
        self.seed = random_seed

        if self.model_type == "extra_trees":
            self.model = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=16,
                min_samples_split=3,
                random_state=self.seed,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            self.model = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=10,
                learning_rate=0.1,
                random_state=self.seed,
            )
        else:  # default random forest
            self.model = RandomForestClassifier(
                n_estimators=150,
                max_depth=14,
                min_samples_split=3,
                random_state=self.seed,
                n_jobs=-1,
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "HighCapacityEnsemble":
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict(X_test)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X_test)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_test, preds)),
        }
