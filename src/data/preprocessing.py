"""
Hyperspectral data preprocessing:
- Spectral calibration using dark and white references (Eq. 3.4)
- Min-Max normalization (Eq. 3.5)
- Pseudo-RGB composite reconstruction (Chapter 4.0.1)
"""

import numpy as np


def calibrate_spectral_cube(raw_cube, dark_ref, white_ref, epsilon=1e-6):
    """
    Performs spectral calibration:
    R_cal = (R - D) / (W - D + epsilon)   [Equation 3.4]
    """
    numerator = raw_cube.astype(np.float32) - dark_ref.astype(np.float32)
    denominator = white_ref.astype(np.float32) - dark_ref.astype(np.float32) + epsilon
    calibrated = numerator / denominator
    return np.clip(calibrated, 0.0, 1.0)


def normalize_min_max(data_cube, epsilon=1e-6):
    """
    Applies min-max normalization:
    X_norm = (X - X_min) / (X_max - X_min + epsilon)   [Equation 3.5]
    """
    x_min = np.min(data_cube)
    x_max = np.max(data_cube)
    return (data_cube - x_min) / (x_max - x_min + epsilon)


def map_band_to_wavelength(band_idx, lambda_min=400.0, delta_lambda=0.5):
    """
    Computes wavelength corresponding to spectral band index b:
    lambda_b = lambda_min + (b - 1) * delta_lambda   [Equation 3.3]
    """
    return lambda_min + (band_idx - 1) * delta_lambda


def generate_pseudo_rgb(hsi_cube, red_band=430, green_band=301, blue_band=189):
    """
    Constructs pseudo-RGB image using 3 representative wavelengths:
    - Red Channel:   Band 430 (~620.00 nm)
    - Green Channel: Band 301 (~553.24 nm)
    - Blue Channel:  Band 189 (~494.57 nm)
    [Report Section 4.0.1]
    """
    h, w, b = hsi_cube.shape
    r = hsi_cube[:, :, min(red_band, b - 1)]
    g = hsi_cube[:, :, min(green_band, b - 1)]
    bl = hsi_cube[:, :, min(blue_band, b - 1)]

    rgb = np.stack([r, g, bl], axis=-1)
    rgb = normalize_min_max(rgb)
    return (rgb * 255.0).astype(np.uint8)
