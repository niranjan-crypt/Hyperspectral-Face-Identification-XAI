"""
Spectral band filter and Signal-to-Noise Ratio (SNR) / dead-band detector.
Explicitly records every retained and excluded band with exact reasons.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class BandExclusionRecord:
    original_index: int
    wavelength_nm: float
    reason: str


class SpectralBandFilter:
    """Filters noisy, uncalibrated, or low-SNR spectral channels."""

    def __init__(
        self,
        min_wavelength_nm: float = 410.0,
        max_wavelength_nm: float = 980.0,
        min_variance: float = 1e-7,
    ):
        self.min_wl = min_wavelength_nm
        self.max_wl = max_wavelength_nm
        self.min_var = min_variance

        self.valid_indices: Optional[np.ndarray] = None
        self.valid_wavelengths: Optional[np.ndarray] = None
        self.exclusions: List[BandExclusionRecord] = []

    def fit(self, wavelengths: np.ndarray, X_train: Optional[np.ndarray] = None) -> "SpectralBandFilter":
        """
        Analyzes bands for edge cutoffs and dead/flat channels.
        Records every exclusion explicitly.
        """
        self.exclusions = []
        valid_idx_list = []

        # Compute variance across training pixels if provided
        var_per_band = None
        if X_train is not None:
            var_per_band = np.var(X_train, axis=0)

        for idx, wl in enumerate(wavelengths):
            if wl < self.min_wl:
                self.exclusions.append(
                    BandExclusionRecord(
                        original_index=idx,
                        wavelength_nm=float(wl),
                        reason=f"Below reliable sensor response threshold (< {self.min_wl:.1f} nm)",
                    )
                )
            elif wl > self.max_wl:
                self.exclusions.append(
                    BandExclusionRecord(
                        original_index=idx,
                        wavelength_nm=float(wl),
                        reason=f"Above reliable NIR detector response threshold (> {self.max_wl:.1f} nm)",
                    )
                )
            elif var_per_band is not None and (var_per_band[idx] < self.min_var or np.isnan(var_per_band[idx])):
                self.exclusions.append(
                    BandExclusionRecord(
                        original_index=idx,
                        wavelength_nm=float(wl),
                        reason=f"Dead or near-zero variance channel (< {self.min_var:.1e})",
                    )
                )
            else:
                valid_idx_list.append(idx)

        self.valid_indices = np.array(valid_idx_list, dtype=np.int32)
        self.valid_wavelengths = wavelengths[self.valid_indices]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Slices array along spectral dimension to retain only valid channels."""
        if self.valid_indices is None:
            raise ValueError("Band filter must be fitted before calling transform!")
        return X[..., self.valid_indices]

    def get_original_index(self, filtered_index: int) -> int:
        """Maps a filtered band index back to the original sensor band index."""
        if self.valid_indices is None:
            raise ValueError("Filter has not been fitted!")
        return int(self.valid_indices[filtered_index])

    def get_original_wavelength(self, filtered_index: int) -> float:
        """Returns the physical wavelength for a filtered band index."""
        if self.valid_wavelengths is None:
            raise ValueError("Filter has not been fitted!")
        return float(self.valid_wavelengths[filtered_index])

    def summary(self) -> str:
        """Returns a formatted summary of excluded and retained channels."""
        total_bands = len(self.valid_indices) + len(self.exclusions)
        retained = len(self.valid_indices)
        excluded = len(self.exclusions)

        lines = [
            "=" * 60,
            "SPECTRAL BAND FILTERING REPORT",
            "=" * 60,
            f"Original sensor bands : {total_bands} (indices 0 to {total_bands - 1})",
            f"Retained active bands : {retained} ({self.valid_wavelengths[0]:.2f} nm to {self.valid_wavelengths[-1]:.2f} nm)",
            f"Excluded noise bands  : {excluded}",
            "-" * 60,
        ]
        # Group exclusions by reason
        reasons = {}
        for rec in self.exclusions:
            reasons.setdefault(rec.reason, []).append(rec.original_index)

        for reason, indices in reasons.items():
            lines.append(f"  [{len(indices)} bands] {reason}: Indices {min(indices)} to {max(indices)}")
        lines.append("=" * 60)
        return "\n".join(lines)
