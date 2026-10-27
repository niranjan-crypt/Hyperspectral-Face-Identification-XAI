"""
Benchmark runner executing equal-budget comparison across multiple random seeds.
Handles JSON checkpointing, held-out test evaluation, and statistical summaries.
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .metrics import ClassificationEvaluator, RedundancyEvaluator
from ..config import AppConfig
from ..data.roi_extractor import FaceDataset
from ..fitness.objective import BandFitnessEvaluator
from ..optimization import (
    BaseOptimizer,
    OptimizationResult,
    PumaOptimizer,
    ParticleSwarmOptimizer,
    WhaleOptimizationAlgorithm,
    GreyWolfOptimizer,
    DifferentialEvolution,
)
from ..preprocessing.normalizer import SpectralNormalizer
from ..preprocessing.snr_filter import SpectralBandFilter


class ExperimentRunner:
    """Orchestrates multi-seed benchmark evaluations with checkpoints."""

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

        # Filter and normalize partitions
        self.X_train = self.normalizer.transform(self.band_filter.transform(dataset.X_train))
        self.X_val = self.normalizer.transform(self.band_filter.transform(dataset.X_val))
        self.X_test = self.normalizer.transform(self.band_filter.transform(dataset.X_test))
        self.y_train = dataset.y_train
        self.y_val = dataset.y_val
        self.y_test = dataset.y_test
        self.wavelengths = band_filter.valid_wavelengths

        # Initialize fitness evaluator (train and val only)
        self.evaluator = BandFitnessEvaluator(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_val,
            y_val=self.y_val,
            w_accuracy=config.fitness.w_accuracy,
            w_redundancy=config.fitness.w_redundancy,
            w_size=config.fitness.w_size,
            metric=config.fitness.metric,
        )
        self.test_evaluator = ClassificationEvaluator(metric=config.fitness.metric)

    def _create_optimizer(self, algo_name: str, seed: int, k_bands: int, pop_size: int, max_iter: int) -> BaseOptimizer:
        total_bands = self.X_train.shape[1]
        name = algo_name.upper()
        if name == "PUMA":
            return PumaOptimizer(k_bands, total_bands, pop_size, max_iter, seed)
        elif name == "PSO":
            return ParticleSwarmOptimizer(k_bands, total_bands, pop_size, max_iter, seed)
        elif name == "WOA":
            return WhaleOptimizationAlgorithm(k_bands, total_bands, pop_size, max_iter, seed)
        elif name == "GWO":
            return GreyWolfOptimizer(k_bands, total_bands, pop_size, max_iter, seed)
        elif name == "DE":
            return DifferentialEvolution(k_bands, total_bands, pop_size, max_iter, seed)
        raise ValueError(f"Unknown optimizer: {algo_name}")

    def run_benchmark(
        self,
        k_bands: Optional[int] = None,
        pop_size: Optional[int] = None,
        max_iter: Optional[int] = None,
        runs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs the full benchmark across all algorithms and random seeds."""
        k = k_bands or self.config.optimization.k_bands
        pop = pop_size or self.config.optimization.pop_size
        iters = max_iter or self.config.optimization.iterations
        seeds = self.config.optimization.random_seeds[:runs] if runs else self.config.optimization.random_seeds
        algorithms = self.config.optimization.algorithms

        os.makedirs(self.config.outputs.checkpoints_dir, exist_ok=True)
        os.makedirs(self.config.outputs.tables_dir, exist_ok=True)

        results_by_algo: Dict[str, List[Dict[str, Any]]] = {algo: [] for algo in algorithms}

        print(f"\n{'='*70}")
        print(f"STARTING BENCHMARK: {len(algorithms)} Algorithms | {len(seeds)} Seeds | K={k} Bands")
        print(f"Budget per run: Pop={pop}, Iterations={iters} -> Max Evals = {pop * (iters + 1):,d}")
        print(f"{'='*70}\n")

        for algo in algorithms:
            print(f"--> Optimizing with {algo}...")
            for seed in seeds:
                ckpt_file = os.path.join(
                    self.config.outputs.checkpoints_dir, f"{algo}_k{k}_seed{seed}.json"
                )

                if os.path.exists(ckpt_file):
                    with open(ckpt_file, "r") as f:
                        run_data = json.load(f)
                    print(f"    [Seed {seed}] Loaded checkpoint (Best Val Fitness: {run_data['best_fitness']:.4f})")
                else:
                    opt = self._create_optimizer(algo, seed, k, pop, iters)
                    res = opt.optimize(self.evaluator.evaluate)

                    # Evaluate on strictly held-out test partition
                    test_metrics = self.test_evaluator.evaluate(
                        self.X_train[:, res.best_bands],
                        self.y_train,
                        self.X_test[:, res.best_bands],
                        self.y_test,
                    )
                    redundancy = RedundancyEvaluator.calculate_redundancy(
                        self.evaluator.corr_matrix, res.best_bands
                    )

                    # Map filtered bands back to original sensor indices and wavelengths
                    original_indices = [self.band_filter.get_original_index(b) for b in res.best_bands]
                    wavelengths = [self.band_filter.get_original_wavelength(b) for b in res.best_bands]

                    run_data = {
                        "algorithm": algo,
                        "seed": int(seed),
                        "k_bands": int(k),
                        "best_fitness": float(res.best_fitness),
                        "convergence": [float(c) for c in res.convergence],
                        "runtime_sec": float(res.runtime_sec),
                        "peak_memory_mb": float(res.peak_memory_mb),
                        "evaluations": int(res.evaluations),
                        "filtered_bands": [int(b) for b in res.best_bands],
                        "original_bands": [int(b) for b in original_indices],
                        "wavelengths": [float(w) for w in wavelengths],
                        "test_accuracy": float(test_metrics["accuracy"]),
                        "test_precision": float(test_metrics["precision"]),
                        "test_recall": float(test_metrics["recall"]),
                        "test_f1": float(test_metrics["f1_macro"]),
                        "test_kappa": float(test_metrics["kappa"]),
                        "redundancy": float(redundancy),
                    }

                    with open(ckpt_file, "w") as f:
                        json.dump(run_data, f, indent=2)

                    print(
                        f"    [Seed {seed}] Done: Fit={res.best_fitness:.4f} | "
                        f"Test OA={test_metrics['accuracy']*100:.2f}% | "
                        f"F1={test_metrics['f1_macro']:.4f} | "
                        f"Time={res.runtime_sec:.2f}s"
                    )

                results_by_algo[algo].append(run_data)

        # Build comparison summary table
        summary_rows = []
        for algo in algorithms:
            runs_list = results_by_algo[algo]
            fitness_vals = [r["best_fitness"] for r in runs_list]
            acc_vals = [r["test_accuracy"] for r in runs_list]
            f1_vals = [r["test_f1"] for r in runs_list]
            kappa_vals = [r["test_kappa"] for r in runs_list]
            red_vals = [r["redundancy"] for r in runs_list]
            times = [r["runtime_sec"] for r in runs_list]
            mems = [r["peak_memory_mb"] for r in runs_list]

            summary_rows.append({
                "Algorithm": algo,
                "Best Fitness": f"{np.min(fitness_vals):.4f}",
                "Mean Fitness": f"{np.mean(fitness_vals):.4f} ± {np.std(fitness_vals):.4f}",
                "Test OA (%)": f"{np.mean(acc_vals)*100:.2f} ± {np.std(acc_vals)*100:.2f}",
                "Test F1-Macro": f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}",
                "Test Kappa": f"{np.mean(kappa_vals):.4f} ± {np.std(kappa_vals):.4f}",
                "Redundancy": f"{np.mean(red_vals):.4f} ± {np.std(red_vals):.4f}",
                "Runtime (s)": f"{np.mean(times):.2f} ± {np.std(times):.2f}",
                "Peak RAM (MB)": f"{np.mean(mems):.2f}",
            })

        csv_path = os.path.join(self.config.outputs.tables_dir, "comparison_summary.csv")
        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

        # Save full raw results
        json_path = os.path.join(self.config.outputs.tables_dir, "comparison_metrics.json")
        with open(json_path, "w") as f:
            json.dump(results_by_algo, f, indent=2)

        print(f"\n{'='*70}")
        print("EXPERIMENT BENCHMARK SUMMARY TABLE")
        print(f"{'='*70}")
        if summary_rows:
            headers = list(summary_rows[0].keys())
            header_line = " | ".join(f"{h:<14}" for h in headers)
            print(header_line)
            print("-" * len(header_line))
            for row in summary_rows:
                print(" | ".join(f"{str(row[h]):<14}" for h in headers))
        print(f"{'='*70}\n")
        print(f"Summary saved to: {csv_path}")

        return results_by_algo
