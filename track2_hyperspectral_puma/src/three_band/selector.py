"""
Three-band optimal subset selector and benchmark comparison.
Determines the best 3-band combination using validation partition,
then evaluates once on the independent test partition.
Compares with:
    1. Winning algorithm's 3-band selection
    2. All-bands baseline
    3. RGB-equivalent bands baseline
    4. Random 3-band baseline (mean of 30 Monte Carlo draws)
"""

import csv
import json
import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from ..config import AppConfig
from ..data.roi_extractor import FaceDataset
from ..evaluation.metrics import ClassificationEvaluator, RedundancyEvaluator
from ..fitness.objective import BandFitnessEvaluator
from ..optimization.puma import PumaOptimizer
from ..preprocessing.normalizer import SpectralNormalizer
from ..preprocessing.snr_filter import SpectralBandFilter


class ThreeBandSelector:
    """Selects and benchmarks the optimal 3-band spectral subset."""

    def __init__(
        self,
        config: AppConfig,
        dataset: FaceDataset,
        band_filter: SpectralBandFilter,
        normalizer: SpectralNormalizer,
    ):
        self.config = config
        self.dataset = dataset
        self.band_filter = band_filter
        self.normalizer = normalizer

        self.X_train = self.normalizer.transform(self.band_filter.transform(dataset.X_train))
        self.X_val = self.normalizer.transform(self.band_filter.transform(dataset.X_val))
        self.X_test = self.normalizer.transform(self.band_filter.transform(dataset.X_test))
        self.y_train = dataset.y_train
        self.y_val = dataset.y_val
        self.y_test = dataset.y_test
        self.wavelengths = band_filter.valid_wavelengths

        self.evaluator = BandFitnessEvaluator(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            w_accuracy=config.fitness.w_accuracy,
            w_redundancy=config.fitness.w_redundancy,
            metric=config.fitness.metric,
        )
        self.test_evaluator = ClassificationEvaluator(metric=config.fitness.metric)

    def _find_nearest_band_idx(self, target_nm: float) -> int:
        return int(np.argmin(np.abs(self.wavelengths - target_nm)))

    def select_best_three_bands(self, pop_size: int = 30, max_iter: int = 50, seed: int = 42) -> Dict[str, Any]:
        """
        Runs dedicated 3-band PUMA optimization and benchmarks against all baselines.
        """
        print(f"\n{'='*70}")
        print("PHASE 9: THREE-BAND SELECTION AND BASELINE BENCHMARKING")
        print(f"{'='*70}")

        total_bands = self.X_train.shape[1]

        # 1. Dedicated PUMA 3-band optimization
        print("[1/4] Running dedicated PUMA 3-band optimization...")
        puma_3 = PumaOptimizer(k_bands=3, total_bands=total_bands, pop_size=pop_size, max_iter=max_iter, random_seed=seed)
        res_puma = puma_3.optimize(self.evaluator.evaluate)
        best_3_filtered = res_puma.best_bands
        best_3_original = [self.band_filter.get_original_index(b) for b in best_3_filtered]
        best_3_wl = [self.band_filter.get_original_wavelength(b) for b in best_3_filtered]

        print(f"    PUMA-3 Best Val Fitness: {res_puma.best_fitness:.4f}")
        print(f"    PUMA-3 Filtered Indices: {best_3_filtered}")
        print(f"    PUMA-3 Original Indices: {best_3_original}")
        print(f"    PUMA-3 Wavelengths (nm): {[round(w, 2) for w in best_3_wl]}")

        # 2. RGB-equivalent bands (~650nm Red, ~550nm Green, ~450nm Blue)
        print("[2/4] Identifying RGB-equivalent spectral channels...")
        rgb_filtered = [
            self._find_nearest_band_idx(650.0),
            self._find_nearest_band_idx(550.0),
            self._find_nearest_band_idx(450.0),
        ]
        rgb_original = [self.band_filter.get_original_index(b) for b in rgb_filtered]
        rgb_wl = [self.band_filter.get_original_wavelength(b) for b in rgb_filtered]
        print(f"    RGB-equivalent Wavelengths: {[round(w, 2) for w in rgb_wl]}")

        # 3. Random 3 bands (mean of 30 Monte Carlo draws)
        print("[3/4] Evaluating Random 3-band Monte Carlo baseline (30 trials)...")
        rng = np.random.RandomState(42)
        rand_test_oas = []
        rand_test_f1s = []
        rand_test_kappas = []
        rand_redundancies = []

        for _ in range(30):
            r_bands = rng.choice(total_bands, size=3, replace=False)
            m = self.test_evaluator.evaluate(
                self.X_train[:, r_bands], self.y_train, self.X_test[:, r_bands], self.y_test
            )
            rand_test_oas.append(m["accuracy"])
            rand_test_f1s.append(m["f1_macro"])
            rand_test_kappas.append(m["kappa"])
            rand_redundancies.append(
                RedundancyEvaluator.calculate_redundancy(self.evaluator.corr_matrix, r_bands)
            )

        # 4. Evaluate all candidates on held-out test partition
        print("[4/4] Conducting test-set validation across all configurations...")

        # A) PUMA-3 bands
        puma_metrics = self.test_evaluator.evaluate(
            self.X_train[:, best_3_filtered], self.y_train, self.X_test[:, best_3_filtered], self.y_test
        )
        puma_red = RedundancyEvaluator.calculate_redundancy(self.evaluator.corr_matrix, best_3_filtered)

        # B) RGB-equivalent bands
        rgb_metrics = self.test_evaluator.evaluate(
            self.X_train[:, rgb_filtered], self.y_train, self.X_test[:, rgb_filtered], self.y_test
        )
        rgb_red = RedundancyEvaluator.calculate_redundancy(self.evaluator.corr_matrix, rgb_filtered)

        # C) All bands (upper bound)
        all_metrics = self.test_evaluator.evaluate(
            self.X_train, self.y_train, self.X_test, self.y_test
        )
        all_red = RedundancyEvaluator.calculate_redundancy(
            self.evaluator.corr_matrix, list(range(total_bands))
        )

        comparison_data = [
            {
                "Configuration": "PO-3 bands (Puma Optimised)",
                "Bands Count": 3,
                "Band Wavelengths (nm)": f"{best_3_wl[0]:.1f}, {best_3_wl[1]:.1f}, {best_3_wl[2]:.1f}",
                "OA": puma_metrics["accuracy"],
                "F1": puma_metrics["f1_macro"],
                "Kappa": puma_metrics["kappa"],
                "Redundancy": puma_red,
            },
            {
                "Configuration": "RGB-equivalent bands",
                "Bands Count": 3,
                "Band Wavelengths (nm)": f"{rgb_wl[0]:.1f}, {rgb_wl[1]:.1f}, {rgb_wl[2]:.1f}",
                "OA": rgb_metrics["accuracy"],
                "F1": rgb_metrics["f1_macro"],
                "Kappa": rgb_metrics["kappa"],
                "Redundancy": rgb_red,
            },
            {
                "Configuration": "Random 3 bands (mean of 30)",
                "Bands Count": 3,
                "Band Wavelengths (nm)": "Random uniform",
                "OA": float(np.mean(rand_test_oas)),
                "F1": float(np.mean(rand_test_f1s)),
                "Kappa": float(np.mean(rand_test_kappas)),
                "Redundancy": float(np.mean(rand_redundancies)),
            },
            {
                "Configuration": "All bands (full spectrum)",
                "Bands Count": total_bands,
                "Band Wavelengths (nm)": f"{self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f}",
                "OA": all_metrics["accuracy"],
                "F1": all_metrics["f1_macro"],
                "Kappa": all_metrics["kappa"],
                "Redundancy": all_red,
            },
        ]

        table_path = os.path.join(self.config.outputs.tables_dir, "three_band_comparison.csv")
        if comparison_data:
            fieldnames = list(comparison_data[0].keys())
            with open(table_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(comparison_data)

        # Print formatted table
        print(f"\n{'='*70}")
        print("THREE-BAND VALIDATION AND BASELINE COMPARISON TABLE")
        print(f"{'='*70}")
        if comparison_data:
            headers = list(comparison_data[0].keys())
            header_line = " | ".join(f"{h:<22}" for h in headers)
            print(header_line)
            print("-" * len(header_line))
            for row in comparison_data:
                print(" | ".join(f"{str(row[h]):<22}" for h in headers))
        print(f"{'='*70}\n")
        print(f"Saved three-band comparison to: {table_path}")

        result_dict = {
            "best_three_bands_filtered": [int(b) for b in best_3_filtered],
            "best_three_bands_original": [int(b) for b in best_3_original],
            "best_three_wavelengths": [float(w) for w in best_3_wl],
            "comparison_table": comparison_data,
        }

        json_path = os.path.join(self.config.outputs.tables_dir, "three_band_winner.json")
        with open(json_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        return result_dict
