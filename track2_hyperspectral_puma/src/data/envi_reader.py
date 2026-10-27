"""
Zero-copy memory-mapped ENVI reader for hyperspectral cubes.
Supports BIL (Band Interleaved by Line), BSQ, and BIP interleaves.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ENVIReader:
    """Reads ENVI format hyperspectral data using memory-mapping."""

    # ENVI data type to NumPy dtype mapping
    DTYPE_MAP = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
        13: np.uint32,
        14: np.int64,
        15: np.uint64,
    }

    def __init__(self, hdr_path: str, spe_path: Optional[str] = None):
        self.hdr_path = hdr_path
        self.metadata = self._parse_header(hdr_path)

        if spe_path is None:
            # Infer from hdr filename
            base, _ = os.path.splitext(hdr_path)
            candidate_spe = base + ".spe"
            if os.path.exists(candidate_spe):
                spe_path = candidate_spe
            elif os.path.exists(base + ".dat"):
                spe_path = base + ".dat"
            else:
                raise FileNotFoundError(f"Could not locate matching binary for {hdr_path}")

        self.spe_path = spe_path
        self.lines: int = int(self.metadata["lines"])
        self.samples: int = int(self.metadata["samples"])
        self.bands: int = int(self.metadata["bands"])
        self.interleave: str = self.metadata.get("interleave", "bil").lower()
        self.byte_order: int = int(self.metadata.get("byte order", 0))
        self.dtype_code: int = int(self.metadata.get("data type", 12))
        self.offset: int = int(self.metadata.get("header offset", 0))
        self.wavelengths: np.ndarray = np.array(self.metadata.get("wavelength", []), dtype=np.float32)

        # Build numpy dtype
        base_dtype = self.DTYPE_MAP.get(self.dtype_code, np.uint16)
        endianness = "<" if self.byte_order == 0 else ">"
        self.dtype = np.dtype(base_dtype).newbyteorder(endianness)

        # Validate file size
        file_size = os.path.getsize(self.spe_path)
        expected_size = self.offset + (self.lines * self.samples * self.bands * self.dtype.itemsize)
        if file_size != expected_size:
            # Note mismatch if any (e.g. if file is float32 instead of uint16)
            if file_size == self.offset + (self.lines * self.samples * self.bands * 4):
                self.dtype = np.dtype(np.float32).newbyteorder(endianness)
            elif file_size == self.offset + (self.lines * self.samples * self.bands * 2):
                self.dtype = np.dtype(np.uint16).newbyteorder(endianness)

        # Memory map layout based on interleave
        if self.interleave == "bil":
            # Band Interleaved by Line: [lines, bands, samples]
            self.shape = (self.lines, self.bands, self.samples)
        elif self.interleave == "bip":
            # Band Interleaved by Pixel: [lines, samples, bands]
            self.shape = (self.lines, self.samples, self.bands)
        elif self.interleave == "bsq":
            # Band Sequential: [bands, lines, samples]
            self.shape = (self.bands, self.lines, self.samples)
        else:
            raise ValueError(f"Unsupported interleave format: {self.interleave}")

        # Open memory-mapped array in read-only mode
        self.mmap: np.memmap = np.memmap(
            self.spe_path,
            dtype=self.dtype,
            mode="r",
            offset=self.offset,
            shape=self.shape,
        )

    def _parse_header(self, hdr_path: str) -> Dict[str, Any]:
        """Parses an ENVI header file (.hdr) into a dictionary."""
        metadata: Dict[str, Any] = {}
        with open(hdr_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extract key-value pairs
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip().lower()
                val = val.strip()

                # Handle multi-line bracketed arrays like wavelength = { ... }
                if val.startswith("{") and not val.endswith("}"):
                    while i + 1 < len(lines) and "}" not in val:
                        i += 1
                        val += " " + lines[i].strip()

                if val.startswith("{") and val.endswith("}"):
                    val_str = val[1:-1].strip()
                    if key in ["wavelength", "default bands", "default thresholds"]:
                        try:
                            val = [float(x.strip()) for x in val_str.split(",") if x.strip()]
                        except ValueError:
                            val = [x.strip() for x in val_str.split(",") if x.strip()]
                    else:
                        val = val_str

                metadata[key] = val
            i += 1
        return metadata

    def get_patch(self, y_min: int, y_max: int, x_min: int, x_max: int) -> np.ndarray:
        """
        Reads a spatial sub-patch across all bands without reading the entire cube.
        Returns array of shape: [height, width, bands]
        """
        if self.interleave == "bil":
            # self.mmap is [lines, bands, samples]
            # slice [y_min:y_max, :, x_min:x_max] -> shape (H, B, W)
            patch = self.mmap[y_min:y_max, :, x_min:x_max]
            # Transpose to (H, W, B)
            return np.transpose(patch, (0, 2, 1))
        elif self.interleave == "bip":
            # self.mmap is [lines, samples, bands]
            return np.array(self.mmap[y_min:y_max, x_min:x_max, :])
        elif self.interleave == "bsq":
            # self.mmap is [bands, lines, samples]
            patch = self.mmap[:, y_min:y_max, x_min:x_max]
            return np.transpose(patch, (1, 2, 0))
        raise ValueError(f"Unknown interleave: {self.interleave}")

    def get_band_image(self, band_idx: int) -> np.ndarray:
        """
        Reads a single spectral band across the full spatial scene.
        Returns array of shape: [lines, samples]
        """
        if self.interleave == "bil":
            return np.array(self.mmap[:, band_idx, :])
        elif self.interleave == "bip":
            return np.array(self.mmap[:, :, band_idx])
        elif self.interleave == "bsq":
            return np.array(self.mmap[band_idx, :, :])
        raise ValueError(f"Unknown interleave: {self.interleave}")

    def get_rgb_preview(self, r_idx: int = 686, g_idx: int = 302, b_idx: int = 187) -> np.ndarray:
        """Constructs a normalized 3-channel RGB image from selected band indices."""
        r = self.get_band_image(r_idx).astype(np.float32)
        g = self.get_band_image(g_idx).astype(np.float32)
        b = self.get_band_image(b_idx).astype(np.float32)

        def norm_channel(ch):
            p2, p98 = np.percentile(ch, (2, 98))
            ch = np.clip(ch, p2, p98)
            rng = p98 - p2
            return (ch - p2) / (rng if rng > 1e-6 else 1.0)

        rgb = np.stack([norm_channel(r), norm_channel(g), norm_channel(b)], axis=-1)
        return rgb
