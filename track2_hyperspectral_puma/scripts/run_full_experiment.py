"""
Full Research Experiment Pipeline:
Runs the complete multi-seed benchmark across all 5 optimizers,
selects the optimal 3-band subset, exports multispectral cubes,
and generates publication-quality figures and comparison tables.
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
from src.three_band.selector import ThreeBandSelector
from src.visualization.plots import Visualizer


def run_pipeline(
    config_path: str = "configs/default_config.yaml",
    k_bands: int = 30,
    pop_size: int = 20,
    iterations: int = 25,
    runs: int = 5,
):
    print("=" * 75)
    print("HYPERSPECTRAL FACE RECOGNITION: PUMA BAND OPTIMIZATION PIPELINE")
    print("=" * 75)
    start_total = time.time()

    cfg = load_config(config_path)

    # 1. Dataset Loading
    print("\n[STEP 1/6] Ingesting and calibrating facial dataset...")
    extractor = FaceDatasetExtractor(cfg)
    dataset = extractor.extract_and_partition()
    print(dataset.summary())

    # 2. Preprocessing & Filtering
    print("\n[STEP 2/6] Applying radiometric band filtering & L2 normalization...")
    band_filter = SpectralBandFilter(
        min_wavelength_nm=cfg.preprocessing.min_wavelength_nm,
        max_wavelength_nm=cfg.preprocessing.max_wavelength_nm,
    )
    band_filter.fit(dataset.wavelengths, dataset.X_train)
    print(band_filter.summary())

    normalizer = SpectralNormalizer(method=cfg.preprocessing.normalizer)
    normalizer.fit(band_filter.transform(dataset.X_train))

    # 3. Optimization Benchmark across 5 Algorithms
    print(f"\n[STEP 3/6] Running Benchmark across 5 Optimizers ({runs} seeds, Pop={pop_size}, Iter={iterations})...")
    runner = ExperimentRunner(cfg, dataset, band_filter, normalizer)
    benchmark_results = runner.run_benchmark(
        k_bands=k_bands,
        pop_size=pop_size,
        max_iter=iterations,
        runs=runs,
    )

    # 4. Three-Band Selection & Baseline Comparison
    print("\n[STEP 4/6] Conducting Dedicated Three-Band Selection & Baseline Comparison...")
    selector = ThreeBandSelector(cfg, dataset, band_filter, normalizer)
    three_band_results = selector.select_best_three_bands(pop_size=pop_size, max_iter=iterations)

    # 5. Multispectral Cube Generation (Full 1920 x 1920 Scene)
    print("\n[STEP 5/6] Exporting Multispectral Cubes and Metadata (.npy and TIFF)...")
    cube_gen = MultispectralCubeGenerator(cfg)

    # Export PUMA best 30-band cube
    puma_runs = benchmark_results["PUMA"]
    best_puma_run = min(puma_runs, key=lambda r: r["best_fitness"])
    cube_gen.generate_multispectral_cube(
        name="puma_best_30bands",
        original_band_indices=best_puma_run["original_bands"],
        wavelengths=best_puma_run["wavelengths"],
    )

    # Export Winner 3-band cube
    best_3_orig = three_band_results["best_three_bands_original"]
    best_3_wl = three_band_results["best_three_wavelengths"]
    cube_gen.generate_multispectral_cube(
        name="po_optimal_3bands",
        original_band_indices=best_3_orig,
        wavelengths=best_3_wl,
    )

    # 6. Scientific Visualizations
    print("\n[STEP 6/6] Generating Publication-Quality Plots...")
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

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f} SECONDS ({total_time/60:.2f} MINUTES)!")
    print(f"Results Directory : {os.path.abspath(cfg.outputs.base_dir)}")
    print(f"Tables Directory  : {os.path.abspath(cfg.outputs.tables_dir)}")
    print(f"Figures Directory : {os.path.abspath(cfg.outputs.figures_dir)}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_pipeline()
