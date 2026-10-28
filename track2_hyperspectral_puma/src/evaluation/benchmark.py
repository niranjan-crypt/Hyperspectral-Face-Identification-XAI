"""
Benchmark runner executing equal-budget 3-band optimization across multiple random seeds.
Performs pure unsupervised band optimization (no classifier training or testing).
Optimizes directly to 3 bands using PUMA and compares with 4 benchmark techniques
(PSO, WOA, GWO, DE) and reference baselines (RGB, Random 3, Full Spectrum).
"""

import csv
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .metrics import RedundancyEvaluator, BandOptimizationEvaluator
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
    """Orchestrates 3-band optimization benchmark evaluations across 5 metaheuristics."""

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

        # Filter and normalize spectral profiles
        self.X_train = self.normalizer.transform(self.band_filter.transform(dataset.X_train))
        self.wavelengths = band_filter.valid_wavelengths

        # Initialize unsupervised fitness evaluator (no classifier training/testing)
        w_red = getattr(config.fitness, "w_redundancy", 0.50)
        w_info = getattr(config.fitness, "w_information", 0.50)
        w_size = getattr(config.fitness, "w_size", 0.00)

        self.evaluator = BandFitnessEvaluator(
            X_train=self.X_train,
            w_redundancy=w_red,
            w_information=w_info,
            w_size=w_size,
            mode="unsupervised",
        )

    def _find_nearest_band_idx(self, target_nm: float) -> int:
        return int(np.argmin(np.abs(self.wavelengths - target_nm)))

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
        k_bands: int = 3,
        pop_size: Optional[int] = None,
        max_iter: Optional[int] = None,
        runs: Optional[int] = None,
        force_recompute: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Runs 3-band optimization across PUMA and 4 benchmark techniques (PSO, WOA, GWO, DE),
        then evaluates against standard reference baselines (RGB, Random, Full Spectrum).
        """
        pop = pop_size or self.config.optimization.pop_size
        iters = max_iter or self.config.optimization.iterations
        seeds = self.config.optimization.random_seeds[:runs] if runs else self.config.optimization.random_seeds
        algorithms = self.config.optimization.algorithms

        os.makedirs(self.config.outputs.checkpoints_dir, exist_ok=True)
        os.makedirs(self.config.outputs.tables_dir, exist_ok=True)

        results_by_algo: Dict[str, List[Dict[str, Any]]] = {algo: [] for algo in algorithms}

        print(f"\n{'='*75}")
        print(f"THREE-BAND OPTIMIZATION BENCHMARK: PUMA vs 4 Competitor Techniques")
        print(f"Algorithms: PUMA (Ours), PSO, WOA, GWO, DE | Target: K = {k_bands} Bands")
        print(f"Mode: Pure Unsupervised (Zero Classification Training/Testing)")
        print(f"Budget per run: Pop={pop}, Iterations={iters} -> Max Evals = {pop * (iters + 1):,d}")
        print(f"{'='*75}\n")

        for algo in algorithms:
            print(f"--> Optimizing 3 bands with {algo}...")
            for seed in seeds:
                ckpt_file = os.path.join(
                    self.config.outputs.checkpoints_dir, f"{algo}_k{k_bands}_seed{seed}.json"
                )

                if not force_recompute and os.path.exists(ckpt_file):
                    with open(ckpt_file, "r") as f:
                        run_data = json.load(f)
                    if "redundancy" in run_data and "information_entropy" in run_data:
                        print(f"    [Seed {seed}] Loaded checkpoint (Best Fitness: {run_data['best_fitness']:.4f})")
                        results_by_algo[algo].append(run_data)
                        continue

                opt = self._create_optimizer(algo, seed, k_bands, pop, iters)
                res = opt.optimize(self.evaluator.evaluate)

                # Evaluate intrinsic spectral properties of the 3 selected bands
                details = self.evaluator.evaluate_detailed(res.best_bands)
                span_nm = RedundancyEvaluator.calculate_spectral_span_nm(self.wavelengths, res.best_bands)

                # Map filtered bands back to original sensor indices and wavelengths
                original_indices = [self.band_filter.get_original_index(b) for b in res.best_bands]
                wavelengths = [self.band_filter.get_original_wavelength(b) for b in res.best_bands]

                run_data = {
                    "algorithm": algo,
                    "seed": int(seed),
                    "k_bands": int(k_bands),
                    "best_fitness": float(res.best_fitness),
                    "convergence": [float(c) for c in res.convergence],
                    "runtime_sec": float(res.runtime_sec),
                    "peak_memory_mb": float(res.peak_memory_mb),
                    "evaluations": int(res.evaluations),
                    "filtered_bands": [int(b) for b in res.best_bands],
                    "original_bands": [int(b) for b in original_indices],
                    "wavelengths": [float(w) for w in wavelengths],
                    "redundancy": float(details["redundancy"]),
                    "information_entropy": float(details["information_entropy"]),
                    "mean_variance": float(details["mean_variance"]),
                    "spectral_span_nm": float(span_nm),
                }

                with open(ckpt_file, "w") as f:
                    json.dump(run_data, f, indent=2)

                print(
                    f"    [Seed {seed}] Done: Fit={res.best_fitness:.4f} | "
                    f"Redundancy={details['redundancy']:.4f} | "
                    f"Entropy={details['information_entropy']:.4f} | "
                    f"Wavelengths={[round(w, 1) for w in wavelengths]} nm | "
                    f"Time={res.runtime_sec:.2f}s"
                )

                results_by_algo[algo].append(run_data)

        # Build comparison summary table across metaheuristics
        summary_rows = []
        for algo in algorithms:
            runs_list = results_by_algo[algo]
            fitness_vals = [r["best_fitness"] for r in runs_list]
            red_vals = [r["redundancy"] for r in runs_list]
            entropy_vals = [r["information_entropy"] for r in runs_list]
            var_vals = [r["mean_variance"] for r in runs_list]
            span_vals = [r["spectral_span_nm"] for r in runs_list]
            times = [r["runtime_sec"] for r in runs_list]
            mems = [r["peak_memory_mb"] for r in runs_list]

            best_run = min(runs_list, key=lambda r: r["best_fitness"])
            best_wls_str = ", ".join(f"{w:.1f}" for w in best_run["wavelengths"])

            summary_rows.append({
                "Algorithm": f"{algo} (Ours)" if algo == "PUMA" else algo,
                "Best 3 Bands (nm)": best_wls_str,
                "Best Fitness": f"{np.min(fitness_vals):.4f}",
                "Mean Fitness": f"{np.mean(fitness_vals):.4f} ± {np.std(fitness_vals):.4f}",
                "Redundancy": f"{np.mean(red_vals):.4f} ± {np.std(red_vals):.4f}",
                "Info Entropy": f"{np.mean(entropy_vals):.4f} ± {np.std(entropy_vals):.4f}",
                "Mean Variance": f"{np.mean(var_vals):.4f} ± {np.std(var_vals):.4f}",
                "Span (nm)": f"{np.mean(span_vals):.1f} ± {np.std(span_vals):.1f}",
                "Runtime (s)": f"{np.mean(times):.2f} ± {np.std(times):.2f}",
                "Peak RAM (MB)": f"{np.mean(mems):.2f}",
            })

        summary_csv = os.path.join(self.config.outputs.tables_dir, "comparison_summary.csv")
        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(summary_rows)

        # Save full raw results JSON
        json_path = os.path.join(self.config.outputs.tables_dir, "comparison_metrics.json")
        with open(json_path, "w") as f:
            json.dump(results_by_algo, f, indent=2)

        # -------------------------------------------------------------
        # REFERENCE BASELINES EVALUATION (RGB, Random 3, Full Spectrum)
        # -------------------------------------------------------------
        total_bands = self.X_train.shape[1]

        # 1. RGB-equivalent bands (~650nm Red, ~550nm Green, ~450nm Blue)
        rgb_filtered = [
            self._find_nearest_band_idx(650.0),
            self._find_nearest_band_idx(550.0),
            self._find_nearest_band_idx(450.0),
        ]
        rgb_wl = [self.band_filter.get_original_wavelength(b) for b in rgb_filtered]
        rgb_details = self.evaluator.evaluate_detailed(rgb_filtered)
        rgb_span = RedundancyEvaluator.calculate_spectral_span_nm(self.wavelengths, rgb_filtered)

        # 2. Random 3 bands (Monte Carlo 30 trials)
        rng = np.random.RandomState(42)
        rand_fitnesses, rand_redundancies, rand_entropies, rand_variances, rand_spans = [], [], [], [], []
        for _ in range(30):
            r_bands = rng.choice(total_bands, size=k_bands, replace=False)
            d = self.evaluator.evaluate_detailed(r_bands)
            rand_fitnesses.append(d["fitness"])
            rand_redundancies.append(d["redundancy"])
            rand_entropies.append(d["information_entropy"])
            rand_variances.append(d["mean_variance"])
            rand_spans.append(RedundancyEvaluator.calculate_spectral_span_nm(self.wavelengths, r_bands))

        # 3. All bands (Full spectrum)
        all_bands = list(range(total_bands))
        all_details = self.evaluator.evaluate_detailed(all_bands)
        all_span = RedundancyEvaluator.calculate_spectral_span_nm(self.wavelengths, all_bands)

        # Build Master 3-Band Comparison Table: PUMA vs 4 Competitor Techniques vs Baselines
        master_three_band_rows = []

        # Add PUMA (Ours)
        best_puma = min(results_by_algo["PUMA"], key=lambda r: r["best_fitness"])
        puma_details = self.evaluator.evaluate_detailed(best_puma["filtered_bands"])
        master_three_band_rows.append({
            "Technique": "PUMA (Ours)",
            "Bands Count": 3,
            "Selected Wavelengths (nm)": ", ".join(f"{w:.1f}" for w in best_puma["wavelengths"]),
            "Fitness (Minimization)": f"{puma_details['fitness']:.4f}",
            "Redundancy": f"{puma_details['redundancy']:.4f}",
            "Info Entropy": f"{puma_details['information_entropy']:.4f}",
            "Mean Variance": f"{puma_details['mean_variance']:.4f}",
            "Span (nm)": f"{best_puma['spectral_span_nm']:.1f}",
        })

        # Add 4 Competitor Metaheuristics (PSO, WOA, GWO, DE)
        for algo in ["PSO", "WOA", "GWO", "DE"]:
            if algo in results_by_algo:
                best_run = min(results_by_algo[algo], key=lambda r: r["best_fitness"])
                d = self.evaluator.evaluate_detailed(best_run["filtered_bands"])
                master_three_band_rows.append({
                    "Technique": f"{algo} (Competitor)",
                    "Bands Count": 3,
                    "Selected Wavelengths (nm)": ", ".join(f"{w:.1f}" for w in best_run["wavelengths"]),
                    "Fitness (Minimization)": f"{d['fitness']:.4f}",
                    "Redundancy": f"{d['redundancy']:.4f}",
                    "Info Entropy": f"{d['information_entropy']:.4f}",
                    "Mean Variance": f"{d['mean_variance']:.4f}",
                    "Span (nm)": f"{best_run['spectral_span_nm']:.1f}",
                })

        # Add Baselines
        master_three_band_rows.extend([
            {
                "Technique": "RGB-equivalent Bands",
                "Bands Count": 3,
                "Selected Wavelengths (nm)": ", ".join(f"{w:.1f}" for w in rgb_wl),
                "Fitness (Minimization)": f"{rgb_details['fitness']:.4f}",
                "Redundancy": f"{rgb_details['redundancy']:.4f}",
                "Info Entropy": f"{rgb_details['information_entropy']:.4f}",
                "Mean Variance": f"{rgb_details['mean_variance']:.4f}",
                "Span (nm)": f"{rgb_span:.1f}",
            },
            {
                "Technique": "Random 3 Bands (Monte Carlo)",
                "Bands Count": 3,
                "Selected Wavelengths (nm)": "Random uniform",
                "Fitness (Minimization)": f"{np.mean(rand_fitnesses):.4f} ± {np.std(rand_fitnesses):.4f}",
                "Redundancy": f"{np.mean(rand_redundancies):.4f} ± {np.std(rand_redundancies):.4f}",
                "Info Entropy": f"{np.mean(rand_entropies):.4f} ± {np.std(rand_entropies):.4f}",
                "Mean Variance": f"{np.mean(rand_variances):.4f} ± {np.std(rand_variances):.4f}",
                "Span (nm)": f"{np.mean(rand_spans):.1f} ± {np.std(rand_spans):.1f}",
            },
            {
                "Technique": "All Bands (Full Spectrum)",
                "Bands Count": total_bands,
                "Selected Wavelengths (nm)": f"{self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f}",
                "Fitness (Minimization)": f"{all_details['fitness']:.4f}",
                "Redundancy": f"{all_details['redundancy']:.4f}",
                "Info Entropy": f"{all_details['information_entropy']:.4f}",
                "Mean Variance": f"{all_details['mean_variance']:.4f}",
                "Span (nm)": f"{all_span:.1f}",
            },
        ])

        three_band_csv = os.path.join(self.config.outputs.tables_dir, "three_band_comparison.csv")
        if master_three_band_rows:
            fnames = list(master_three_band_rows[0].keys())
            with open(three_band_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fnames)
                writer.writeheader()
                writer.writerows(master_three_band_rows)

        # Save winner JSON
        winner_dict = {
            "best_three_bands_filtered": [int(b) for b in best_puma["filtered_bands"]],
            "best_three_bands_original": [int(b) for b in best_puma["original_bands"]],
            "best_three_wavelengths": [float(w) for w in best_puma["wavelengths"]],
            "puma_fitness": float(best_puma["best_fitness"]),
            "puma_redundancy": float(best_puma["redundancy"]),
            "puma_information_entropy": float(best_puma["information_entropy"]),
            "comparison_table": master_three_band_rows,
        }
        winner_json = os.path.join(self.config.outputs.tables_dir, "three_band_winner.json")
        with open(winner_json, "w") as f:
            json.dump(winner_dict, f, indent=2)

        # Print formatted master table
        print(f"\n{'='*80}")
        print("MASTER 3-BAND OPTIMIZATION COMPARISON TABLE")
        print(f"{'='*80}")
        if master_three_band_rows:
            headers = list(master_three_band_rows[0].keys())
            header_line = " | ".join(f"{h:<26}" for h in headers)
            print(header_line)
            print("-" * len(header_line))
            for row in master_three_band_rows:
                print(" | ".join(f"{str(row[h]):<26}" for h in headers))
        print(f"{'='*80}\n")
        print(f"Summary saved to: {summary_csv}")
        print(f"3-Band comparison saved to: {three_band_csv}")

        return results_by_algo, winner_dict
