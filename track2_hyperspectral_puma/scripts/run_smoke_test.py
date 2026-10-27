"""
Smoke test script for verifying all 5 metaheuristic band selection algorithms.
Runs 2 iterations with population size 6 in ~15-20 seconds.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.data import FaceDatasetExtractor
from src.preprocessing import SpectralBandFilter, SpectralNormalizer
from src.fitness import BandFitnessEvaluator
from src.optimization import (
    PumaOptimizer,
    ParticleSwarmOptimizer,
    WhaleOptimizationAlgorithm,
    GreyWolfOptimizer,
    DifferentialEvolution,
)


def run_smoke_test():
    print("=" * 65)
    print("STARTING RAPID SMOKE TEST: ALL 5 OPTIMIZATION ALGORITHMS")
    print("=" * 65)

    cfg = load_config("configs/default_config.yaml")

    # 1. Load data
    print("[1/4] Loading face dataset...")
    extractor = FaceDatasetExtractor(cfg)
    dataset = extractor.extract_and_partition()

    # 2. Filter & Normalize
    print("[2/4] Applying band filtering and normalization...")
    band_filter = SpectralBandFilter(min_wavelength_nm=410.0, max_wavelength_nm=980.0)
    band_filter.fit(dataset.wavelengths, dataset.X_train)

    X_train_f = band_filter.transform(dataset.X_train)
    X_val_f = band_filter.transform(dataset.X_val)

    normalizer = SpectralNormalizer(method="l2").fit(X_train_f)
    X_train_norm = normalizer.transform(X_train_f)
    X_val_norm = normalizer.transform(X_val_f)

    # 3. Setup Fitness Evaluator
    print("[3/4] Initializing multi-objective fitness evaluator...")
    evaluator = BandFitnessEvaluator(
        X_train=X_train_norm,
        y_train=dataset.y_train,
        X_val=X_val_norm,
        y_val=dataset.y_val,
        w_accuracy=0.80,
        w_redundancy=0.20,
    )

    # 4. Smoke Test all 5 algorithms
    total_bands = X_train_norm.shape[1]
    k_bands = 10
    pop_size = 6
    max_iter = 2
    seed = 42

    optimizers = [
        PumaOptimizer(k_bands=k_bands, total_bands=total_bands, pop_size=pop_size, max_iter=max_iter, random_seed=seed),
        ParticleSwarmOptimizer(k_bands=k_bands, total_bands=total_bands, pop_size=pop_size, max_iter=max_iter, random_seed=seed),
        WhaleOptimizationAlgorithm(k_bands=k_bands, total_bands=total_bands, pop_size=pop_size, max_iter=max_iter, random_seed=seed),
        GreyWolfOptimizer(k_bands=k_bands, total_bands=total_bands, pop_size=pop_size, max_iter=max_iter, random_seed=seed),
        DifferentialEvolution(k_bands=k_bands, total_bands=total_bands, pop_size=pop_size, max_iter=max_iter, random_seed=seed),
    ]

    print("\n[4/4] Running 2 iterations per algorithm:")
    print(f"{'Algorithm':<10} | {'Best Fitness':<12} | {'Val Acc (%)':<12} | {'Redundancy':<12} | {'Time (s)':<10} | {'RAM (MB)':<10}")
    print("-" * 75)

    all_passed = True
    for opt in optimizers:
        t0 = time.time()
        res = opt.optimize(evaluator.evaluate)
        t_cost = time.time() - t0

        # Detailed breakdown
        details = evaluator.evaluate_detailed(res.best_bands)

        # Check unique constraint
        is_unique = len(res.best_bands) == len(set(res.best_bands)) == k_bands
        if not is_unique:
            print(f"[ERROR] {opt.name} violated uniqueness constraint!")
            all_passed = False

        print(
            f"{opt.name:<10} | {res.best_fitness:<12.4f} | {details['accuracy_val']*100:<12.2f} | "
            f"{details['redundancy']:<12.4f} | {t_cost:<10.2f} | {res.peak_memory_mb:<10.2f}"
        )

    print("=" * 75)
    if all_passed:
        print("[SUCCESS] All 5 optimization algorithms completed smoke test successfully!")
    else:
        print("[FAILED] One or more algorithms failed constraints.")
    print("=" * 75)


if __name__ == "__main__":
    run_smoke_test()
