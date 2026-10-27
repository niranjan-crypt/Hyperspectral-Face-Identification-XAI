"""
Multi-objective fitness evaluator for hyperspectral spectral band selection.
Optimisation Direction: MINIMIZATION (Lower score indicates better performance).
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from ..features.classifier import FastNearestCentroid


class BandFitnessEvaluator:
    """
    Evaluates candidate spectral band subsets using:
        f(S) = w_1 * (1 - Acc_val) + w_2 * Redundancy(S) + w_3 * (|S| / B)

    Direction: MINIMIZATION (Lower is better: higher accuracy, lower redundancy).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        w_accuracy: float = 0.80,
        w_redundancy: float = 0.20,
        w_size: float = 0.00,
        metric: str = "cosine",
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.w_acc = w_accuracy
        self.w_red = w_redundancy
        self.w_size = w_size
        self.metric = metric
        self.total_bands = X_train.shape[1]

        # Precompute band-to-band correlation matrix for microsecond redundancy calculations
        # Subsample pixels if training set is large to compute correlation matrix instantly
        n_samples = min(len(X_train), 10000)
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(X_train), size=n_samples, replace=False)
        X_sub = X_train[sub_idx]

        # Pearson correlation matrix: [B, B]
        std_devs = np.std(X_sub, axis=0, keepdims=True)
        std_devs[std_devs < 1e-6] = 1.0
        X_centered = (X_sub - np.mean(X_sub, axis=0, keepdims=True)) / std_devs
        self.corr_matrix = np.abs(np.dot(X_centered.T, X_centered) / n_samples).astype(np.float32)
        np.fill_diagonal(self.corr_matrix, 1.0)

    def evaluate(self, candidate_bands: Sequence[int]) -> float:
        """
        Calculates the scalar fitness value for a candidate band subset.
        Returns float value in range [0, 1] where lower is better.
        """
        bands = list(dict.fromkeys(int(b) for b in candidate_bands if 0 <= b < self.total_bands))
        if not bands:
            return 1.0  # Maximum penalty for empty or completely invalid subset

        # 1. Classification Error on Validation Partition
        clf = FastNearestCentroid(metric=self.metric)
        clf.fit(self.X_train[:, bands], self.y_train)
        acc_val = clf.score(self.X_val[:, bands], self.y_val)
        error_val = 1.0 - acc_val

        # 2. Inter-Band Redundancy (Average Pairwise Absolute Correlation)
        k = len(bands)
        if k <= 1:
            redundancy = 0.0
        else:
            sub_corr = self.corr_matrix[np.ix_(bands, bands)]
            # Sum off-diagonal elements
            triu_indices = np.triu_indices(k, k=1)
            redundancy = float(np.mean(sub_corr[triu_indices]))

        # 3. Size Penalty
        size_ratio = k / self.total_bands

        # Scalarized fitness (minimization)
        fitness = (self.w_acc * error_val) + (self.w_red * redundancy) + (self.w_size * size_ratio)
        return float(fitness)

    def evaluate_detailed(self, candidate_bands: Sequence[int]) -> Dict[str, float]:
        """Provides full breakdown of accuracy, redundancy, and fitness components."""
        bands = list(dict.fromkeys(int(b) for b in candidate_bands if 0 <= b < self.total_bands))
        if not bands:
            return {"fitness": 1.0, "accuracy_val": 0.0, "error_val": 1.0, "redundancy": 1.0, "num_bands": 0}

        clf = FastNearestCentroid(metric=self.metric)
        clf.fit(self.X_train[:, bands], self.y_train)
        acc_val = clf.score(self.X_val[:, bands], self.y_val)
        error_val = 1.0 - acc_val

        k = len(bands)
        if k <= 1:
            redundancy = 0.0
        else:
            sub_corr = self.corr_matrix[np.ix_(bands, bands)]
            triu_indices = np.triu_indices(k, k=1)
            redundancy = float(np.mean(sub_corr[triu_indices]))

        size_ratio = k / self.total_bands
        fitness = (self.w_acc * error_val) + (self.w_red * redundancy) + (self.w_size * size_ratio)

        return {
            "fitness": float(fitness),
            "accuracy_val": float(acc_val),
            "error_val": float(error_val),
            "redundancy": float(redundancy),
            "num_bands": k,
        }
