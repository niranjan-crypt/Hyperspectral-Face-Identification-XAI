"""
Multi-objective fitness evaluator for hyperspectral spectral band selection.
Optimisation Direction: MINIMIZATION (Lower score indicates better performance).

Default Mode: UNSUPERVISED (No training/testing classifiers, evaluates intrinsic
spectral information entropy, variance, and inter-band redundancy).
"""

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class BandFitnessEvaluator:
    """
    Evaluates candidate spectral band subsets without classifier training or testing.
    Uses information-theoretic and spectral metrics:
        f(S) = w_red * Redundancy(S) + w_info * (1 - Information(S)) + w_size * (|S| / B)

    Direction: MINIMIZATION (Lower is better: minimal redundancy, maximum information).
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        w_redundancy: float = 0.50,
        w_information: float = 0.50,
        w_accuracy: float = 0.00,
        w_size: float = 0.00,
        metric: str = "cosine",
        mode: str = "unsupervised",
        num_entropy_bins: int = 50,
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.w_red = w_redundancy
        self.w_info = w_information
        self.w_acc = w_accuracy
        self.w_size = w_size
        self.metric = metric
        self.mode = mode
        self.num_bins = num_entropy_bins
        self.total_bands = X_train.shape[1]

        # Subsample pixels to precompute statistics instantly
        n_samples = min(len(X_train), 10000)
        rng = np.random.RandomState(42)
        sub_idx = rng.choice(len(X_train), size=n_samples, replace=False)
        X_sub = X_train[sub_idx]

        # 1. Pearson correlation matrix: [B, B]
        std_devs = np.std(X_sub, axis=0, keepdims=True)
        std_devs[std_devs < 1e-6] = 1.0
        X_centered = (X_sub - np.mean(X_sub, axis=0, keepdims=True)) / std_devs
        self.corr_matrix = np.abs(np.dot(X_centered.T, X_centered) / n_samples).astype(np.float32)
        np.fill_diagonal(self.corr_matrix, 1.0)

        # 2. Per-band normalized Shannon entropy: [B]
        self.entropies = np.zeros(self.total_bands, dtype=np.float32)
        max_h = float(np.log2(self.num_bins))
        for b in range(self.total_bands):
            counts, _ = np.histogram(X_sub[:, b], bins=self.num_bins)
            p = counts.astype(np.float32) / float(np.sum(counts))
            p = p[p > 0]
            self.entropies[b] = float(-np.sum(p * np.log2(p)) / max_h)

        # 3. Per-band normalized standard deviation / variance: [B]
        raw_stds = np.std(X_sub, axis=0)
        max_std = float(np.max(raw_stds))
        self.variances = (raw_stds / (max_std + 1e-7)).astype(np.float32)

        # Combined per-channel information capacity: [B]
        self.channel_info = 0.6 * self.entropies + 0.4 * self.variances

    def evaluate(self, candidate_bands: Sequence[int]) -> float:
        """
        Calculates the scalar fitness value for a candidate band subset.
        Returns float value in range [0, 1] where lower is better.
        No classifier training or testing is performed.
        """
        bands = list(dict.fromkeys(int(b) for b in candidate_bands if 0 <= b < self.total_bands))
        if not bands:
            return 1.0  # Maximum penalty for empty or completely invalid subset

        k = len(bands)

        # 1. Inter-Band Redundancy (Average Pairwise Absolute Correlation)
        if k <= 1:
            redundancy = 0.0
        else:
            sub_corr = self.corr_matrix[np.ix_(bands, bands)]
            triu_indices = np.triu_indices(k, k=1)
            redundancy = float(np.mean(sub_corr[triu_indices]))

        # 2. Information Content (Normalized Entropy & Variance)
        info_content = float(np.mean(self.channel_info[bands]))
        info_deficit = 1.0 - info_content

        # 3. Size Penalty
        size_ratio = k / self.total_bands

        fitness = (self.w_red * redundancy) + (self.w_info * info_deficit) + (self.w_size * size_ratio)
        return float(fitness)

    def evaluate_detailed(self, candidate_bands: Sequence[int]) -> Dict[str, float]:
        """Provides full breakdown of redundancy, entropy, variance, and fitness components."""
        bands = list(dict.fromkeys(int(b) for b in candidate_bands if 0 <= b < self.total_bands))
        if not bands:
            return {
                "fitness": 1.0,
                "redundancy": 1.0,
                "information_entropy": 0.0,
                "mean_variance": 0.0,
                "information_content": 0.0,
                "num_bands": 0,
            }

        k = len(bands)
        if k <= 1:
            redundancy = 0.0
        else:
            sub_corr = self.corr_matrix[np.ix_(bands, bands)]
            triu_indices = np.triu_indices(k, k=1)
            redundancy = float(np.mean(sub_corr[triu_indices]))

        mean_entropy = float(np.mean(self.entropies[bands]))
        mean_var = float(np.mean(self.variances[bands]))
        info_content = float(np.mean(self.channel_info[bands]))
        info_deficit = 1.0 - info_content
        size_ratio = k / self.total_bands

        fitness = (self.w_red * redundancy) + (self.w_info * info_deficit) + (self.w_size * size_ratio)

        return {
            "fitness": float(fitness),
            "redundancy": float(redundancy),
            "information_entropy": float(mean_entropy),
            "mean_variance": float(mean_var),
            "information_content": float(info_content),
            "num_bands": k,
        }
