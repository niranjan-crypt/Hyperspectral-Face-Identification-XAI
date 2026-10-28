"""
Spectral feature normalizer (L2, Standard, MinMax).
Normalizes pixel reflectance profiles across active spectral channels.
"""

from typing import Optional
import numpy as np


class SpectralNormalizer:
    """Normalizes spectral vectors along channel dimension."""

    def __init__(self, method: str = "l2"):
        self.method = method.lower()
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        if self.method == "standard":
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            self.std_ = np.std(X, axis=0, keepdims=True)
            self.std_[self.std_ < 1e-8] = 1.0
        elif self.method == "minmax":
            self.min_ = np.min(X, axis=0, keepdims=True)
            self.max_ = np.max(X, axis=0, keepdims=True)
            denom = self.max_ - self.min_
            denom[denom < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.method == "l2":
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms < 1e-8] = 1.0
            return (X / norms).astype(np.float32)
        elif self.method == "standard":
            if self.mean_ is None or self.std_ is None:
                self.fit(X)
            return ((X - self.mean_) / self.std_).astype(np.float32)
        elif self.method == "minmax":
            if self.min_ is None or self.max_ is None:
                self.fit(X)
            denom = self.max_ - self.min_
            denom[denom < 1e-8] = 1.0
            return ((X - self.min_) / denom).astype(np.float32)
        return X.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

