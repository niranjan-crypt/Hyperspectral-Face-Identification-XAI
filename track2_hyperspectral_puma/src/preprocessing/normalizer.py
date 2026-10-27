"""
Spectral normalization module.
Supports L2 vector normalization (per-pixel illumination invariance),
standard scaling, and min-max normalization fitted strictly on training data.
"""

from typing import Optional, Union
import numpy as np


class SpectralNormalizer:
    """Normalizes spectral vectors while preventing data leakage."""

    def __init__(self, method: str = "l2"):
        self.method = method.lower()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray) -> "SpectralNormalizer":
        """Fits normalization parameters strictly on training partition."""
        if self.method == "standard":
            self.mean = np.mean(X_train, axis=0, keepdims=True)
            self.std = np.std(X_train, axis=0, keepdims=True)
            self.std[self.std < 1e-6] = 1.0
        elif self.method == "minmax":
            self.min_val = np.min(X_train, axis=0, keepdims=True)
            self.max_val = np.max(X_train, axis=0, keepdims=True)
            diff = self.max_val - self.min_val
            diff[diff < 1e-6] = 1.0
            self.max_val = self.min_val + diff
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms input spectral array."""
        if self.method == "l2":
            # Per-pixel L2 spectral normalization: removes illumination and shadow variations
            norms = np.linalg.norm(X, axis=-1, keepdims=True)
            norms[norms < 1e-7] = 1.0
            return (X / norms).astype(np.float32)
        elif self.method == "standard":
            if self.mean is None or self.std is None:
                raise ValueError("Normalizer must be fitted on X_train before transform!")
            return ((X - self.mean) / self.std).astype(np.float32)
        elif self.method == "minmax":
            if self.min_val is None or self.max_val is None:
                raise ValueError("Normalizer must be fitted on X_train before transform!")
            diff = self.max_val - self.min_val
            return ((X - self.min_val) / diff).astype(np.float32)
        else:
            return X.astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit and transform training set."""
        return self.fit(X_train).transform(X_train)
