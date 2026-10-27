"""
Multispectral cube extraction and export module.
Extracts selected spectral channels from the original hyperspectral acquisition,
preserving full spatial dimensions (1920 x 1920) and original channel mappings.
Saves data as NumPy arrays (.npy), multi-channel TIFFs, and comprehensive JSON metadata.
"""

import json
import os
from typing import Dict, List, Optional, Sequence
import numpy as np
from PIL import Image

from ..config import AppConfig
from ..data.envi_reader import ENVIReader


class MultispectralCubeGenerator:
    """Extracts and saves multispectral cubes for selected spectral band subsets."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reader = ENVIReader(
            hdr_path=config.dataset.get_path("hdr_file"),
            spe_path=config.dataset.get_path("raw_spe"),
        )
        self.out_dir = config.outputs.multispectral_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def generate_multispectral_cube(
        self,
        name: str,
        original_band_indices: Sequence[int],
        wavelengths: Sequence[float],
    ) -> Dict[str, str]:
        """
        Extracts selected spectral channels for the full 1920 x 1920 scene.
        Memory requirement for 3 bands: ~22 MB. For 30 bands: ~220 MB.
        """
        k = len(original_band_indices)
        h, w = self.reader.lines, self.reader.samples
        print(f"--> Generating multispectral cube '{name}' ({k} channels, {w}x{h})...")

        # Extract channels into a compact array [lines, samples, k]
        cube = np.zeros((h, w, k), dtype=np.uint16)
        for i, b_idx in enumerate(original_band_indices):
            cube[:, :, i] = self.reader.get_band_image(int(b_idx))

        # 1. Save as NumPy .npy file
        npy_path = os.path.join(self.out_dir, f"{name}_multispectral.npy")
        np.save(npy_path, cube)

        # 2. Save as multi-frame / multi-channel TIFF
        tiff_path = os.path.join(self.out_dir, f"{name}_multispectral.tiff")
        if k == 3:
            # Normalize to 8-bit for standard viewing compatibility while preserving dynamic range
            ch_list = []
            for i in range(3):
                ch = cube[:, :, i].astype(np.float32)
                p2, p98 = np.percentile(ch, (1, 99))
                ch_norm = np.clip((ch - p2) / (p98 - p2 + 1e-6) * 255.0, 0, 255).astype(np.uint8)
                ch_list.append(Image.fromarray(ch_norm))
            tiff_img = Image.merge("RGB", tuple(ch_list))
            tiff_img.save(tiff_path)
        else:
            # Multi-page TIFF for K > 3
            frames = [Image.fromarray((cube[:, :, i] >> 4).astype(np.uint8)) for i in range(k)]
            frames[0].save(tiff_path, save_all=True, append_images=frames[1:])

        # 3. Save JSON metadata
        meta_path = os.path.join(self.out_dir, f"{name}_metadata.json")
        metadata = {
            "name": name,
            "spatial_dimensions": {"lines": h, "samples": w},
            "num_selected_bands": k,
            "original_band_indices": [int(x) for x in original_band_indices],
            "wavelengths_nm": [float(x) for x in wavelengths],
            "data_type": "uint16",
            "file_npy": os.path.basename(npy_path),
            "file_tiff": os.path.basename(tiff_path),
            "scientific_note": (
                "Band selection extracts a discrete subset of existing hyperspectral channels. "
                "This reduces data bandwidth and redundant dimensionality, but is mathematically "
                "distinct from simulating or manufacturing a physical multispectral sensor with "
                "broad spectral response filter curves."
            ),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"    Saved NPY  : {npy_path} ({os.path.getsize(npy_path)/(1024**2):.2f} MB)")
        print(f"    Saved TIFF : {tiff_path} ({os.path.getsize(tiff_path)/(1024**2):.2f} MB)")
        print(f"    Saved Meta : {meta_path}")

        return {
            "npy_path": npy_path,
            "tiff_path": tiff_path,
            "meta_path": meta_path,
        }
