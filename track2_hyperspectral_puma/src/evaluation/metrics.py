"""
Comprehensive scientific evaluation metrics:
Overall Accuracy (OA), Macro Precision, Macro Recall, Macro F1, Cohen's Kappa,
and spectral redundancy / diversity metrics.
"""

from typing import Dict, List, Sequence
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

from ..features.classifier import FastNearestCentroid


class ClassificationEvaluator:
    """Computes comprehensive classification metrics on held-out test partitions."""

    def __init__(self, metric: str = "cosine"):
        self.metric = metric

    def evaluate(
        self,
        X_train_sub: np.ndarray,
        y_train: np.ndarray,
        X_test_sub: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        clf = FastNearestCentroid(metric=self.metric)
        clf.fit(X_train_sub, y_train)
        preds = clf.predict(X_test_sub)

        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, average="macro", zero_division=0)),
            "recall": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y_test, preds)),
        }


class RedundancyEvaluator:
    """Measures spectral redundancy and diversity of selected band subsets."""

    @staticmethod
    def calculate_redundancy(corr_matrix: np.ndarray, bands: Sequence[int]) -> float:
        k = len(bands)
        if k <= 1:
            return 0.0
        sub_corr = corr_matrix[np.ix_(bands, bands)]
        triu = np.triu_indices(k, k=1)
        return float(np.mean(sub_corr[triu]))

    @staticmethod
    def calculate_spectral_span_nm(wavelengths: np.ndarray, bands: Sequence[int]) -> float:
        wl_sub = wavelengths[list(bands)]
        return float(np.max(wl_sub) - np.min(wl_sub))
