"""
Face ROI extractor and spatial block dataset partitioner.
Extracts facial regions for the 7 subjects, calibrates them, and partitions them
into disjoint train, validation, and test sets to prevent spatial data leakage.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .envi_reader import ENVIReader
from .calibration import RadiometricCalibrator
from ..config import AppConfig


@dataclass
class FaceDataset:
    X_train: np.ndarray  # [N_train, bands]
    y_train: np.ndarray  # [N_train]
    X_val: np.ndarray    # [N_val, bands]
    y_val: np.ndarray    # [N_val]
    X_test: np.ndarray   # [N_test, bands]
    y_test: np.ndarray   # [N_test]
    wavelengths: np.ndarray
    subject_names: List[str]

    def summary(self) -> str:
        train_sz = self.X_train.nbytes / (1024**2)
        val_sz = self.X_val.nbytes / (1024**2)
        test_sz = self.X_test.nbytes / (1024**2)
        total_sz = train_sz + val_sz + test_sz
        lines = [
            "=" * 60,
            "FACE DATASET SUMMARY",
            "=" * 60,
            f"Subjects (Classes) : {len(self.subject_names)}",
            f"Spectral Channels  : {self.X_train.shape[1]} bands ({self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm)",
            f"Training samples   : {len(self.y_train):,d} pixels ({train_sz:.2f} MB)",
            f"Validation samples : {len(self.y_val):,d} pixels ({val_sz:.2f} MB)",
            f"Test samples       : {len(self.y_test):,d} pixels ({test_sz:.2f} MB)",
            f"Total cached RAM   : {total_sz:.2f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)


class FaceDatasetExtractor:
    """Extracts, calibrates, and partitions face ROIs."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.reader = ENVIReader(
            hdr_path=config.dataset.get_path("hdr_file"),
            spe_path=config.dataset.get_path("raw_spe"),
        )
        self.calibrator = RadiometricCalibrator(
            white_path=config.dataset.get_path("figspec_white"),
            black_path=config.dataset.get_path("figspec_black"),
            excel_path=config.dataset.get_path("calibration_table"),
            num_bands=config.geometry.bands,
            samples=config.geometry.samples,
        )

    def extract_and_partition(self, force_recompute: bool = False) -> FaceDataset:
        """
        Extracts face ROIs, applies radiometric calibration, and partitions into
        spatially disjoint train/val/test splits. Caches result to NPZ for speed.
        """
        cache_path = os.path.join(self.config.outputs.cache_dir, "face_dataset.npz")
        if not force_recompute and os.path.exists(cache_path):
            data = np.load(cache_path)
            return FaceDataset(
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_val=data["X_val"],
                y_val=data["y_val"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                wavelengths=data["wavelengths"],
                subject_names=list(data["subject_names"]),
            )

        rng = np.random.RandomState(self.config.split.random_seed)
        block_size = self.config.split.spatial_block_size
        train_ratio = self.config.split.train_ratio
        val_ratio = self.config.split.val_ratio

        train_X_list, train_y_list = [], []
        val_X_list, val_y_list = [], []
        test_X_list, test_y_list = [], []
        subject_names = []

        for sub_idx, (name, box) in enumerate(self.config.rois.items()):
            subject_names.append(name)
            ymin, ymax, xmin, xmax = box

            # 1. Read raw DN patch from memmap [H, W, B]
            raw_patch = self.reader.get_patch(ymin, ymax, xmin, xmax)

            # 2. Calibrate patch to reflectance [H, W, B]
            calib_patch = self.calibrator.calibrate(raw_patch)
            h, w, b = calib_patch.shape

            # 3. Partition into disjoint spatial blocks
            blocks = []
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    block_pixels = calib_patch[y : y + block_size, x : x + block_size, :]
                    flat_pixels = block_pixels.reshape(-1, b)
                    if len(flat_pixels) > 0:
                        blocks.append(flat_pixels)

            # Shuffle blocks reproducibly
            block_indices = np.arange(len(blocks))
            rng.shuffle(block_indices)

            n_train = max(1, int(len(blocks) * train_ratio))
            n_val = max(1, int(len(blocks) * val_ratio))

            train_blocks = [blocks[i] for i in block_indices[:n_train]]
            val_blocks = [blocks[i] for i in block_indices[n_train : n_train + n_val]]
            test_blocks = [blocks[i] for i in block_indices[n_train + n_val :]]

            if not test_blocks:
                test_blocks = [val_blocks[-1]]

            for blk in train_blocks:
                train_X_list.append(blk)
                train_y_list.append(np.full(len(blk), sub_idx, dtype=np.int32))

            for blk in val_blocks:
                val_X_list.append(blk)
                val_y_list.append(np.full(len(blk), sub_idx, dtype=np.int32))

            for blk in test_blocks:
                test_X_list.append(blk)
                test_y_list.append(np.full(len(blk), sub_idx, dtype=np.int32))

        X_train = np.vstack(train_X_list).astype(np.float32)
        y_train = np.concatenate(train_y_list).astype(np.int32)
        X_val = np.vstack(val_X_list).astype(np.float32)
        y_val = np.concatenate(val_y_list).astype(np.int32)
        X_test = np.vstack(test_X_list).astype(np.float32)
        y_test = np.concatenate(test_y_list).astype(np.int32)

        dataset = FaceDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            wavelengths=self.reader.wavelengths,
            subject_names=subject_names,
        )

        # Save cache
        np.savez_compressed(
            cache_path,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            wavelengths=self.reader.wavelengths,
            subject_names=np.array(subject_names),
        )

        return dataset
