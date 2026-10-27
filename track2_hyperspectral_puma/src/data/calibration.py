"""
Radiometric calibration module for FigSpec hyperspectral imaging systems.
Calibrates raw Digital Numbers (DN) to spectral reflectance.
"""

import os
import struct
from typing import Optional, Tuple
import numpy as np


class RadiometricCalibrator:
    """
    Performs radiometric calibration using FigSpec white and black reference files:
        R(lambda) = [(DN_raw - DN_dark) / (DN_white - DN_dark)] * R_standard(lambda)
    """

    def __init__(
        self,
        white_path: str,
        black_path: str,
        excel_path: Optional[str] = None,
        num_bands: int = 1200,
        samples: int = 1920,
    ):
        self.white_path = white_path
        self.black_path = black_path
        self.excel_path = excel_path
        self.num_bands = num_bands
        self.samples = samples

        self.wavelengths, self.r_standard, self.white_dn = self._parse_white_file(white_path)
        self.dark_dn_profile = self._parse_black_file(black_path)

        # Precompute denominator: white_dn - dark_dn
        # white_dn is [B], dark_dn_profile is [B] (or [S, B])
        self.denominator = self.white_dn.astype(np.float32) - self.dark_dn_profile.astype(np.float32)
        # Avoid division by zero
        self.denominator[np.abs(self.denominator) < 1.0] = 1.0

    def _parse_white_file(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parses figspecwhite binary to extract wavelengths, standard reflectance, and white DN."""
        with open(path, "rb") as f:
            buf = f.read()

        # Tag 12: Wavelengths (1200 float32 = 4800 bytes)
        # Tag 13: Standard Reflectance (1200 float32 = 4800 bytes)
        # Tag 14: White Reference DN (1200 uint16 = 2400 bytes)
        wavelengths = None
        r_standard = None
        white_dn = None

        offset = 0
        while offset < len(buf) - 10:
            tag, data_len = struct.unpack("<HQ", buf[offset:offset + 10])
            payload = buf[offset + 10:offset + 10 + data_len]
            if tag == 12 and data_len == self.num_bands * 4:
                wavelengths = np.frombuffer(payload, dtype="<f4")
            elif tag == 13 and data_len == self.num_bands * 4:
                r_standard = np.frombuffer(payload, dtype="<f4")
            elif tag == 14 and data_len == self.num_bands * 2:
                white_dn = np.frombuffer(payload, dtype="<u2")
            offset += 10 + data_len

        # Fallbacks if parsing fails
        if wavelengths is None:
            wavelengths = np.linspace(393.77, 1014.73, self.num_bands, dtype=np.float32)
        if r_standard is None:
            r_standard = np.full(self.num_bands, 0.95, dtype=np.float32)
        if white_dn is None:
            # Synthetic white level fallback
            white_dn = np.full(self.num_bands, 25000, dtype=np.uint16)

        return wavelengths, r_standard, white_dn

    def _parse_black_file(self, path: str) -> np.ndarray:
        """Parses figspecblack binary to extract dark current baseline."""
        with open(path, "rb") as f:
            buf = f.read()

        dark_profile = None
        offset = 0
        while offset < len(buf) - 10:
            tag, data_len = struct.unpack("<HQ", buf[offset:offset + 10])
            payload = buf[offset + 10:offset + 10 + data_len]
            # Dark frame is often 1920 samples * 1200 bands * 2 bytes = 4,608,000 bytes
            if data_len == self.samples * self.num_bands * 2:
                # Shape in BIL is [bands, samples]
                dark_2d = np.frombuffer(payload, dtype="<u2").reshape((self.num_bands, self.samples))
                # Mean over spatial dimension -> [B]
                dark_profile = np.mean(dark_2d, axis=1)
                break
            elif data_len == self.num_bands * 2:
                dark_profile = np.frombuffer(payload, dtype="<u2").astype(np.float32)
                break
            offset += 10 + data_len

        if dark_profile is None:
            dark_profile = np.full(self.num_bands, 0.0, dtype=np.float32)

        return dark_profile

    def calibrate(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Calibrates raw DN array to surface reflectance.
        Can process:
            - 1D spectrum: [bands]
            - 2D spectra: [N_pixels, bands]
            - 3D cube: [height, width, bands]
        """
        raw_float = raw_data.astype(np.float32)
        # raw_float shape has bands at the last dimension
        numerator = raw_float - self.dark_dn_profile
        reflectance = (numerator / self.denominator) * self.r_standard
        # Physically realistic reflectance: non-negative, clipped at upper bound 2.0
        return np.clip(reflectance, 0.0, 2.0)
