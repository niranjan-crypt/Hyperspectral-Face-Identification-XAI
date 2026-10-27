"""
Multi-Scale Retinex (MSR) Illumination Normalization Module.
Decomposes spatial-spectral reflectance from non-uniform ambient illumination,
compensating for lighting shadows across peripheral subjects in group captures.
"""

from typing import List, Optional, Sequence
import numpy as np
from scipy.ndimage import gaussian_filter


class MultiScaleRetinex:
    """
    Applies Multi-Scale Retinex (MSR) with dynamic range compression
    to eliminate illumination falloff and shadow gradients.
    """

    def __init__(
        self,
        scales: Sequence[float] = (5.0, 15.0, 40.0),
        weights: Optional[Sequence[float]] = None,
        dynamic_clip_percentiles: tuple = (1.0, 99.0),
    ):
        self.scales = scales
        if weights is None:
            self.weights = [1.0 / len(scales)] * len(scales)
        else:
            self.weights = [w / sum(weights) for w in weights]
        self.clip_low, self.clip_high = dynamic_clip_percentiles

    def transform_channel(self, img_2d: np.ndarray) -> np.ndarray:
        """Processes a single 2D spatial channel with MSR."""
        eps = 1e-6
        img_float = np.maximum(img_2d.astype(np.float32), eps)
        log_img = np.log(img_float)

        retinex = np.zeros_like(img_float)
        for scale, weight in zip(self.scales, self.weights):
            # Estimate illumination via Gaussian blur
            illumination = gaussian_filter(img_float, sigma=scale)
            log_illum = np.log(np.maximum(illumination, eps))
            retinex += weight * (log_img - log_illum)

        # Dynamic range contrast normalization
        v_min = np.percentile(retinex, self.clip_low)
        v_max = np.percentile(retinex, self.clip_high)
        if v_max > v_min:
            retinex = np.clip((retinex - v_min) / (v_max - v_min), 0.0, 1.0)
        else:
            retinex = np.zeros_like(retinex)

        return retinex.astype(np.float32)

    def transform_cube(self, cube_3d: np.ndarray) -> np.ndarray:
        """
        Processes a 3D spatial-spectral cube [Height, Width, Channels].
        Applies MSR channel-by-channel across the spectral bands.
        """
        h, w, c = cube_3d.shape
        out = np.zeros((h, w, c), dtype=np.float32)
        for b in range(c):
            out[:, :, b] = self.transform_channel(cube_3d[:, :, b])
        return out
