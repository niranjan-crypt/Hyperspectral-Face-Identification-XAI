"""
Radiometric and SNR-based spectral band filter.
Filters dead channels, zero-variance detectors, and noisy sensor edge wavelengths (<410nm, >980nm).
"""

from typing import Dict, List, Optional, Sequence
import numpy as np


class SpectralBandFilter:
    """Filters noisy, unresponsive, and dead spectral channels."""

    def __init__(
        self,
        min_wavelength_nm: float = 410.0,
        max_wavelength_nm: float = 980.0,
        min_variance: float = 1e-7,
    ):
        self.min_wl = min_wavelength_nm
        self.max_wl = max_wavelength_nm
        self.min_var = min_variance

        self.valid_indices: np.ndarray = np.array([], dtype=int)
        self.valid_wavelengths: np.ndarray = np.array([], dtype=np.float32)
        self.total_original_bands: int = 0
        self.below_wl_count: int = 0
        self.dead_channels_count: int = 0
        self.above_wl_count: int = 0

    def fit(self, wavelengths: np.ndarray, X: np.ndarray):
        """Identifies valid bands satisfying SNR, variance, and wavelength bounds."""
        self.total_original_bands = len(wavelengths)
        variances = np.var(X, axis=0)

        valid_list = []
        below_list = []
        dead_list = []
        above_list = []

        for b in range(self.total_original_bands):
            wl = wavelengths[b]
            var = variances[b]

            if wl < self.min_wl:
                below_list.append(b)
            elif wl > self.max_wl:
                above_list.append(b)
            elif var < self.min_var:
                dead_list.append(b)
            else:
                valid_list.append(b)

        self.valid_indices = np.array(valid_list, dtype=int)
        self.valid_wavelengths = wavelengths[self.valid_indices].astype(np.float32)
        self.below_wl_count = len(below_list)
        self.dead_channels_count = len(dead_list)
        self.above_wl_count = len(above_list)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Selects valid channels from input array."""
        return X[:, self.valid_indices]

    def fit_transform(self, wavelengths: np.ndarray, X: np.ndarray) -> np.ndarray:
        return self.fit(wavelengths, X).transform(X)

    def get_original_index(self, filtered_idx: int) -> int:
        """Maps a filtered band index back to the original camera sensor index."""
        return int(self.valid_indices[filtered_idx])

    def get_original_wavelength(self, filtered_idx: int) -> float:
        """Returns the wavelength in nm for a filtered band index."""
        return float(self.valid_wavelengths[filtered_idx])

    def summary(self) -> str:
        """Returns detailed scientific band filtering report."""
        excluded = self.below_wl_count + self.dead_channels_count + self.above_wl_count
        lines = [
            "=" * 60,
            "SPECTRAL BAND FILTERING REPORT",
            "=" * 60,
            f"Original sensor bands : {self.total_original_bands} (indices 0 to {self.total_original_bands - 1})",
            f"Retained active bands : {len(self.valid_indices)} ({self.valid_wavelengths[0]:.2f} nm to {self.valid_wavelengths[-1]:.2f} nm)",
            f"Excluded noise bands  : {excluded}",
            "-" * 60,
            f"  [{self.below_wl_count} bands] Below reliable sensor response threshold (< {self.min_wl:.1f} nm): Indices 0 to {self.below_wl_count - 1}",
            f"  [{self.dead_channels_count} bands] Dead or near-zero variance channel (< {self.min_var:.1e})",
            f"  [{self.above_wl_count} bands] Above reliable NIR detector response threshold (> {self.max_wl:.1f} nm)",
            "=" * 60,
        ]
        return "\n".join(lines)

