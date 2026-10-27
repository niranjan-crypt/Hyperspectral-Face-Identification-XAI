"""
Spatial-Spectral Feature Extractor and Patch Aggregator.
Combines local spatial neighborhood statistics (mean, variance, inter-quantile contrast)
with physiologically grounded spectral ratios (NDSEI, Melanin-Hemoglobin Index).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class SpatialSpectralFeatureExtractor:
    """Extracts enriched spatial-spectral descriptors from local facial blocks."""

    def __init__(self, patch_size: int = 16, include_ratios: bool = True):
        self.patch_size = patch_size
        self.include_ratios = include_ratios

    def aggregate_patches(
        self,
        X_pixels: np.ndarray,
        y_pixels: np.ndarray,
        patch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregates sequential contiguous spatial blocks into patch-level descriptors.
        Since pixels within disjoint blocks belong to the same subject, grouping
        by patch_size creates noise-reduced regional feature vectors.
        """
        ps = patch_size or self.patch_size
        n_total = len(X_pixels)
        n_patches = n_total // ps

        if n_patches == 0:
            return X_pixels.mean(axis=0, keepdims=True), np.array([y_pixels[0]])

        # Reshape into [N_patches, patch_size, n_bands]
        truncated_len = n_patches * ps
        X_blocks = X_pixels[:truncated_len].reshape(n_patches, ps, -1)
        y_blocks = y_pixels[:truncated_len:ps]

        # 1. Primary spatial mean (noise reduction)
        patch_mean = np.mean(X_blocks, axis=1)

        # 2. Local spatial texture variance
        patch_std = np.std(X_blocks, axis=1)

        # 3. Inter-quantile spatial contrast (p90 - p10)
        p10 = np.percentile(X_blocks, 10, axis=1)
        p90 = np.percentile(X_blocks, 90, axis=1)
        patch_contrast = p90 - p10

        feature_list = [patch_mean, patch_std, patch_contrast]

        # 4. Spectral physiological indices if 3 or more bands are present
        if self.include_ratios and X_pixels.shape[1] >= 3:
            b_hb = patch_mean[:, 0]
            b_mel = patch_mean[:, 1]
            b_nir = patch_mean[:, 2]

            eps = 1e-6
            # Normalized Difference Skin Erythema Index: (NIR - Hb) / (NIR + Hb)
            ndsei = ((b_nir - b_hb) / (b_nir + b_hb + eps))[:, None]
            # Melanin-Hemoglobin Contrast Ratio: Mel / (Hb + eps)
            mhr = (b_mel / (b_hb + eps))[:, None]
            # Tissue Scattering Index: NIR / (Mel + eps)
            tsi = (b_nir / (b_mel + eps))[:, None]
            # Relative Spectral Slopes
            slope_vis = (b_mel - b_hb)[:, None]
            slope_nir = (b_nir - b_mel)[:, None]

            indices = np.column_stack([ndsei, mhr, tsi, slope_vis, slope_nir])
            feature_list.append(indices)

        X_features = np.column_stack(feature_list).astype(np.float32)
        return X_features, y_blocks.astype(np.int32)
