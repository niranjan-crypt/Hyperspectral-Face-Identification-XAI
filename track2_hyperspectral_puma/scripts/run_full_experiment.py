"""
Full Research Experiment Pipeline:
Takes 4 benchmark band optimization techniques (PSO, WOA, GWO, DE), minimizes directly
to 3 bands, and compares them against our PUMA optimization.
Mode: Pure Unsupervised Band Optimization (No Classifier Training or Testing).
Exports optimal 3-band multispectral cubes and generates publication-quality figures.
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import load_config
from src.data import FaceDatasetExtractor
from src.evaluation.benchmark import ExperimentRunner
from src.multispectral.cube_generator import MultispectralCubeGenerator
from src.preprocessing.normalizer import SpectralNormalizer
from src.preprocessing.snr_filter import SpectralBandFilter
from src.visualization.plots import Visualizer


def run_pipeline(
    config_path: str = "configs/default_config.yaml",
    k_bands: int = 3,
    pop_size: int = 25,
    iterations: int = 30,
    runs: int = 5,
    force_recompute: bool = True,
):
    print("=" * 80)
    print("HYPERSPECTRAL 3-BAND OPTIMIZATION: PUMA vs 4 BENCHMARK TECHNIQUES")
    print("Target: Directly Minimize to 3 Optimal Bands (Zero Classification / Testing)")
    print("=" * 80)
    start_total = time.time()

    cfg = load_config(config_path)

    # 1. Dataset Ingestion & Calibration
    print("\n[STEP 1/5] Ingesting and calibrating facial dataset (1200 channels)...")
    extractor = FaceDatasetExtractor(cfg)
    dataset = extractor.extract_and_partition()
    print(dataset.summary())

    # 2. Preprocessing & Filtering
    print("\n[STEP 2/5] Applying radiometric band filtering & L2 normalization...")
    band_filter = SpectralBandFilter(
        min_wavelength_nm=cfg.preprocessing.min_wavelength_nm,
        max_wavelength_nm=cfg.preprocessing.max_wavelength_nm,
    )
    band_filter.fit(dataset.wavelengths, dataset.X_train)
    print(band_filter.summary())

    normalizer = SpectralNormalizer(method=cfg.preprocessing.normalizer)
    normalizer.fit(band_filter.transform(dataset.X_train))

    # 3. 3-Band Optimization Benchmark across PUMA and 4 Competitor Techniques
    print(f"\n[STEP 3/5] Minimizing to 3 Bands across PUMA, PSO, WOA, GWO, DE ({runs} runs, Pop={pop_size}, Iter={iterations})...")
    runner = ExperimentRunner(cfg, dataset, band_filter, normalizer)
    benchmark_results, winner_dict = runner.run_benchmark(
        k_bands=k_bands,
        pop_size=pop_size,
        max_iter=iterations,
        runs=runs,
        force_recompute=force_recompute,
    )

    # 4. Multispectral Cube Generation (Winning PUMA 3-Band Scene: 1920 x 1920)
    print("\n[STEP 4/5] Exporting Optimal 3-Band Multispectral Cube (.npy and TIFF)...")
    cube_gen = MultispectralCubeGenerator(cfg)

    best_3_orig = winner_dict["best_three_bands_original"]
    best_3_wl = winner_dict["best_three_wavelengths"]
    cube_gen.generate_multispectral_cube(
        name="po_optimal_3bands",
        original_band_indices=best_3_orig,
        wavelengths=best_3_wl,
    )

    # 5. Scientific Visualizations
    print("\n[STEP 5/5] Generating Publication-Quality Plots for 3-Band Optimization...")
    viz = Visualizer(output_dir=cfg.outputs.figures_dir)
    f_conv = viz.plot_convergence(benchmark_results)
    print(f"    Saved: {f_conv}")
    f_box = viz.plot_metrics_comparison(benchmark_results)
    print(f"    Saved: {f_box}")
    f_hist = viz.plot_band_distribution(benchmark_results)
    print(f"    Saved: {f_hist}")

    X_train_f = band_filter.transform(dataset.X_train)
    f_sig = viz.plot_spectral_signatures(
        wavelengths=band_filter.valid_wavelengths,
        X_train=X_train_f,
        y_train=dataset.y_train,
        subject_names=dataset.subject_names,
        selected_3_wavelengths=best_3_wl,
    )
    print(f"    Saved: {f_sig}")

    f_corr = viz.plot_correlation_heatmap(
        full_corr=runner.evaluator.corr_matrix,
        selected_bands=winner_dict["best_three_bands_filtered"],
        selected_wavelengths=best_3_wl,
    )
    print(f"    Saved: {f_corr}")

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"3-BAND OPTIMIZATION COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS ({total_time/60:.2f} MINUTES)!")
    print(f"Results Directory : {os.path.abspath(cfg.outputs.base_dir)}")
    print(f"Tables Directory  : {os.path.abspath(cfg.outputs.tables_dir)}")
    print(f"Figures Directory : {os.path.abspath(cfg.outputs.figures_dir)}")
    print("=" * 80 + "\n")

    return benchmark_results, winner_dict


if __name__ == "__main__":
    run_pipeline()
