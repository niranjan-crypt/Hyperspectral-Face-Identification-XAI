"""
Configuration management with dataclass typing and YAML loading.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
import yaml


@dataclass
class DatasetConfig:
    base_dir: str
    raw_spe: str
    hdr_file: str
    ref_spe: str
    figspec_white: str
    figspec_black: str
    calibration_table: str
    preview_jpg: str

    def get_path(self, key: str) -> str:
        rel = getattr(self, key)
        candidate = os.path.join(self.base_dir, rel)
        if os.path.exists(candidate):
            return candidate
        # Check fallback search locations (current dir, parent dir, data folder, package root)
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        repo_root = os.path.dirname(package_root)
        for d in [".", "..", "data", os.path.join("..", "data"), repo_root, package_root]:
            p = os.path.join(d, rel)
            if os.path.exists(p):
                return p
        return candidate


@dataclass
class GeometryConfig:
    samples: int
    lines: int
    bands: int
    interleave: str
    byte_order: int
    dtype: str


@dataclass
class SplitConfig:
    train_ratio: float
    val_ratio: float
    test_ratio: float
    spatial_block_size: int
    random_seed: int


@dataclass
class PreprocessingConfig:
    normalizer: str
    remove_noisy_edges: bool
    min_wavelength_nm: float
    max_wavelength_nm: float


@dataclass
class FitnessConfig:
    w_accuracy: float
    w_redundancy: float
    w_size: float
    classifier: str
    metric: str


@dataclass
class OptimizationConfig:
    k_bands: int
    final_target_bands: int
    pop_size: int
    iterations: int
    runs: int
    random_seeds: List[int]
    algorithms: List[str]


@dataclass
class OutputsConfig:
    base_dir: str
    checkpoints_dir: str
    figures_dir: str
    tables_dir: str
    multispectral_dir: str
    cache_dir: str


@dataclass
class AppConfig:
    dataset: DatasetConfig
    geometry: GeometryConfig
    rois: Dict[str, List[int]]
    split: SplitConfig
    preprocessing: PreprocessingConfig
    fitness: FitnessConfig
    optimization: OptimizationConfig
    outputs: OutputsConfig


def load_config(config_path: str = "configs/default_config.yaml") -> AppConfig:
    """Load and parse YAML configuration into strongly-typed AppConfig."""
    if not os.path.exists(config_path):
        pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(pkg_root, config_path)
        if os.path.exists(alt):
            config_path = alt
        elif os.path.exists(os.path.join(os.path.dirname(pkg_root), config_path)):
            config_path = os.path.join(os.path.dirname(pkg_root), config_path)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = AppConfig(
        dataset=DatasetConfig(**data["dataset"]),
        geometry=GeometryConfig(**data["geometry"]),
        rois=data["rois"],
        split=SplitConfig(**data["split"]),
        preprocessing=PreprocessingConfig(**data["preprocessing"]),
        fitness=FitnessConfig(**data["fitness"]),
        optimization=OptimizationConfig(**data["optimization"]),
        outputs=OutputsConfig(**data["outputs"]),
    )

    # Ensure output directories exist
    for dir_path in [
        cfg.outputs.base_dir,
        cfg.outputs.checkpoints_dir,
        cfg.outputs.figures_dir,
        cfg.outputs.tables_dir,
        cfg.outputs.multispectral_dir,
        cfg.outputs.cache_dir,
    ]:
        os.makedirs(dir_path, exist_ok=True)

    return cfg
