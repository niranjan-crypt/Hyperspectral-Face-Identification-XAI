"""
Comprehensive scientific evaluation metrics:
Focuses on intrinsic spectral band optimization metrics (redundancy, information entropy,
spectral span, dynamic range) without requiring classifier training or testing.
"""

from typing import Dict, List, Sequence, Optional
import numpy as np


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


class BandOptimizationEvaluator:
    """Evaluates the intrinsic quality of selected spectral bands without classifier training/testing."""

    @staticmethod
    def evaluate_bands(
        corr_matrix: np.ndarray,
        entropies: np.ndarray,
        variances: np.ndarray,
        wavelengths: np.ndarray,
        bands: Sequence[int],
    ) -> Dict[str, float]:
        k = len(bands)
        if k == 0:
            return {
                "redundancy": 1.0,
                "information_entropy": 0.0,
                "mean_variance": 0.0,
                "spectral_span_nm": 0.0,
                "num_bands": 0,
            }

        redundancy = RedundancyEvaluator.calculate_redundancy(corr_matrix, bands)
        span_nm = RedundancyEvaluator.calculate_spectral_span_nm(wavelengths, bands)
        mean_entropy = float(np.mean(entropies[list(bands)]))
        mean_var = float(np.mean(variances[list(bands)]))

        return {
            "redundancy": float(redundancy),
            "information_entropy": float(mean_entropy),
            "mean_variance": float(mean_var),
            "spectral_span_nm": float(span_nm),
            "num_bands": k,
        }



