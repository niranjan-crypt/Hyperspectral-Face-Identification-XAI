"""
Lightweight and fast classifiers for hyperspectral human face discrimination.
Includes vectorized FastNearestCentroid for microsecond optimization loops
and LinearSVMClassifier for benchmark validation.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_score, recall_score


class FastNearestCentroid:
    """
    Ultra-fast vectorized Nearest Centroid classifier using Cosine distance.
    Evaluates 40,000 pixels in ~1-2 milliseconds on CPU.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric
        self.centroids: Optional[np.ndarray] = None  # [num_classes, num_bands]
        self.classes: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FastNearestCentroid":
        """Computes class centroids from training data."""
        self.classes = np.unique(y)
        num_classes = len(self.classes)
        num_bands = X.shape[1]
        self.centroids = np.zeros((num_classes, num_bands), dtype=np.float32)

        for c_idx, c in enumerate(self.classes):
            mask = (y == c)
            self.centroids[c_idx] = np.mean(X[mask], axis=0)

        if self.metric == "cosine":
            # Normalize centroids to unit length for fast cosine inner product
            norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
            norms[norms < 1e-7] = 1.0
            self.centroids = self.centroids / norms
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction."""
        if self.centroids is None or self.classes is None:
            raise ValueError("Classifier is not fitted.")

        if self.metric == "cosine":
            # X must be normalized or normalized on the fly
            X_norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_norms[X_norms < 1e-7] = 1.0
            X_unit = X / X_norms
            # Similarity matrix: [N_samples, num_classes]
            sims = np.dot(X_unit, self.centroids.T)
            best_idx = np.argmax(sims, axis=1)
            return self.classes[best_idx]
        else:
            # Euclidean distance: ||x - mu||^2
            # dist = ||x||^2 - 2 * x . mu^T + ||mu||^2
            x_sq = np.sum(X**2, axis=1, keepdims=True)
            mu_sq = np.sum(self.centroids**2, axis=1, keepdims=True).T
            cross = 2.0 * np.dot(X, self.centroids.T)
            dists = x_sq - cross + mu_sq
            best_idx = np.argmin(dists, axis=1)
            return self.classes[best_idx]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Returns classification accuracy."""
        preds = self.predict(X)
        return float(np.mean(preds == y))


class LinearSVMClassifier:
    """Linear Support Vector Classifier for thorough benchmark verification."""

    def __init__(self, c_param: float = 1.0, max_iter: int = 1000):
        self.model = LinearSVC(C=c_param, max_iter=max_iter, random_state=42, dual=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def evaluate_metrics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, average="macro", zero_division=0)),
            "recall": float(recall_score(y, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(y, preds)),
        }
