"""
Master Proof Runner for Hyperspectral Face Recognition with PUMA.
Executes:
    1. Peer-reviewed Statistical Hypothesis Tests (Wilcoxon Signed-Rank with Holm-Bonferroni, Friedman Ranking)
    2. Multi-dimensional Fisher Scatter & Matrix Condition Number & Mutual Information
    3. Cross-Classifier Generalization Benchmarks (Nearest Centroid, Linear SVM, Random Forest, k-NN)
    4. 2D Manifold Cluster Separability (t-SNE, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz)
    5. Publication Figures (t-SNE Separability, 7x7 Confusion Matrix, Radar Benchmark Chart, Cross-Classifier Bar Chart)
"""

import csv
import json
import os
import sys
os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs/cache/matplotlib")
from typing import Dict, List, Any

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.config import load_config
from src.preprocessing.snr_filter import SpectralBandFilter
from src.preprocessing.normalizer import SpectralNormalizer
from src.evaluation.statistical_proofs import StatisticalProofEngine
from src.evaluation.multi_classifier import MultiClassifierBenchmarker
from src.evaluation.manifold import ManifoldClusterEvaluator
from src.visualization.advanced_plots import AdvancedVisualizer
from src.features.classifier import FastNearestCentroid


def main():
    print("=" * 80)
    print("ADVANCED PROOF ENGINE: HYPERSPECTRAL FACE RECOGNITION (PUMA)")
    print("=" * 80)

    cfg = load_config("configs/default_config.yaml")

    # 1. Load cached dataset
    cache_path = os.path.abspath("outputs/cache/face_dataset.npz")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cached dataset not found at {cache_path}. Run full experiment first.")

    print(f"\n[1/6] Loading cached facial dataset from {cache_path}...")
    npz = np.load(cache_path, allow_pickle=True)
    X_train_raw = npz["X_train"]
    y_train = npz["y_train"]
    X_val_raw = npz["X_val"]
    y_val = npz["y_val"]
    X_test_raw = npz["X_test"]
    y_test = npz["y_test"]
    raw_wavelengths = npz["wavelengths"]
    subject_names = [str(s) for s in npz["subject_names"]]

    print(f"      Train raw: {X_train_raw.shape}, Val raw: {X_val_raw.shape}, Test raw: {X_test_raw.shape}")
    print(f"      Subjects ({len(subject_names)}): {subject_names}")

    # 2. Filter noisy/dead edge bands (1200 -> 925 channels)
    print("\n[2/6] Applying radiometric band filtering & L2 spectral normalization...")
    band_filter = SpectralBandFilter(
        min_wavelength_nm=cfg.preprocessing.min_wavelength_nm,
        max_wavelength_nm=cfg.preprocessing.max_wavelength_nm,
    )
    band_filter.fit(raw_wavelengths, X_train_raw)
    valid_wavelengths = band_filter.valid_wavelengths

    X_train_filt = band_filter.transform(X_train_raw)
    X_val_filt = band_filter.transform(X_val_raw)
    X_test_filt = band_filter.transform(X_test_raw)
    print(f"      Retained active channels: {X_train_filt.shape[1]} ({valid_wavelengths[0]:.1f} - {valid_wavelengths[-1]:.1f} nm)")

    # 3. L2 per-pixel illumination-invariant normalization
    normalizer = SpectralNormalizer(method="l2")
    X_tr_norm = normalizer.transform(X_train_filt)
    X_va_norm = normalizer.transform(X_val_filt)
    X_te_norm = normalizer.transform(X_test_filt)

    # 4. Load existing experiment results & winning 3 bands
    comp_json_path = os.path.abspath("outputs/tables/comparison_metrics.json")
    winner_json_path = os.path.abspath("outputs/tables/three_band_winner.json")

    with open(comp_json_path, "r") as f:
        comp_metrics = json.load(f)

    with open(winner_json_path, "r") as f:
        winner_data = json.load(f)

    best_3_filtered = winner_data["best_three_bands_filtered"]
    best_3_wl = winner_data["best_three_wavelengths"]
    print(f"      Loaded winning PUMA 3-band subset: {best_3_filtered} ({[round(w, 1) for w in best_3_wl]} nm)")

    # Find RGB-equivalent bands
    def find_nearest_band(nm_target):
        return int(np.argmin(np.abs(valid_wavelengths - nm_target)))

    rgb_filtered = [find_nearest_band(650.0), find_nearest_band(550.0), find_nearest_band(450.0)]
    rgb_wl = [float(valid_wavelengths[b]) for b in rgb_filtered]
    print(f"      RGB-equivalent bands: {rgb_filtered} ({[round(w, 1) for w in rgb_wl]} nm)")

    # Sample a fixed deterministic random 3-band subset for reproducibility
    rng = np.random.RandomState(42)
    rand_3_filtered = sorted(rng.choice(X_tr_norm.shape[1], size=3, replace=False).tolist())
    rand_3_wl = [float(valid_wavelengths[b]) for b in rand_3_filtered]
    print(f"      Random 3-band baseline: {rand_3_filtered} ({[round(w, 1) for w in rand_3_wl]} nm)")

    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # 5. STATISTICAL HYPOTHESIS TESTING
    print("\n" + "=" * 80)
    print("PHASE A: STATISTICAL HYPOTHESIS TESTS (Wilcoxon & Friedman)")
    print("=" * 80)

    puma_fitness = [r["best_fitness"] for r in comp_metrics["PUMA"]]
    puma_accuracy = [r["test_accuracy"] for r in comp_metrics["PUMA"]]

    competitor_fitness = {a: [r["best_fitness"] for r in comp_metrics[a]] for a in ["PSO", "WOA", "GWO", "DE"]}
    competitor_accuracy = {a: [r["test_accuracy"] for r in comp_metrics[a]] for a in ["PSO", "WOA", "GWO", "DE"]}

    wilcoxon_fit = StatisticalProofEngine.compute_wilcoxon_tests(puma_fitness, competitor_fitness, alternative="less")
    wilcoxon_acc = StatisticalProofEngine.compute_wilcoxon_tests(puma_accuracy, competitor_accuracy, alternative="greater")

    all_fitness = {"PUMA": puma_fitness, **competitor_fitness}
    all_acc = {"PUMA": puma_accuracy, **competitor_accuracy}
    friedman_fit = StatisticalProofEngine.compute_friedman_test(all_fitness, minimize=True)
    friedman_acc = StatisticalProofEngine.compute_friedman_test(all_acc, minimize=False)

    print("\n--- Pairwise Wilcoxon Signed-Rank Test (PUMA vs Competitors on Fitness) ---")
    for comp, res in wilcoxon_fit.items():
        print(f"  PUMA vs {comp:4s}: W-stat = {res['statistic']:5.1f}, raw p = {res['raw_p']:.4f}, Holm-Adj p = {res['adjusted_p']:.4f} (Sig@0.05: {res['significant_05']})")

    print("\n--- Friedman Multi-Algorithm Rankings (Fitness Minimization) ---")
    print(f"  Chi-Square Stat = {friedman_fit['friedman_stat']:.4f}, p-value = {friedman_fit['p_value']:.4e}")
    for algo, rank in friedman_fit["average_ranks"].items():
        print(f"  Rank {rank:.2f}: {algo}")

    # Save Wilcoxon results to CSV
    wilcoxon_csv_path = "outputs/tables/statistical_tests_wilcoxon.csv"
    with open(wilcoxon_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Comparison", "Metric", "W_Statistic", "Raw_p_value", "Holm_Adjusted_p", "Significant_Alpha_0.05"])
        for comp, res in wilcoxon_fit.items():
            writer.writerow([f"PUMA vs {comp}", "Objective Fitness", f"{res['statistic']:.2f}", f"{res['raw_p']:.5f}", f"{res['adjusted_p']:.5f}", res["significant_05"]])
        for comp, res in wilcoxon_acc.items():
            writer.writerow([f"PUMA vs {comp}", "Test Accuracy", f"{res['statistic']:.2f}", f"{res['raw_p']:.5f}", f"{res['adjusted_p']:.5f}", res["significant_05"]])
    print(f"  Saved Wilcoxon tests -> {wilcoxon_csv_path}")

    # Save Friedman results to JSON
    friedman_json_path = "outputs/tables/friedman_rankings.json"
    with open(friedman_json_path, "w") as f:
        json.dump({"fitness_minimization": friedman_fit, "accuracy_maximization": friedman_acc}, f, indent=2)
    print(f"  Saved Friedman rankings -> {friedman_json_path}")

    # 6. FISHER MULTIVARIATE SCATTER & CONDITION NUMBER & MUTUAL INFORMATION
    print("\n" + "=" * 80)
    print("PHASE B: INFORMATION THEORY & MULTIVARIATE SEPARABILITY PROOFS")
    print("=" * 80)

    subsets = {
        "PO-3 (Puma Optimised)": best_3_filtered,
        "RGB-equivalent bands": rgb_filtered,
        "Random 3 bands": rand_3_filtered,
        "All 925 bands (Full Spectrum)": list(range(X_te_norm.shape[1])),
    }

    info_results = []
    for name, b_indices in subsets.items():
        print(f"  Computing mathematical metrics for: {name} ({len(b_indices)} bands)...")
        J_fisher = StatisticalProofEngine.compute_fisher_multivariate_criterion(X_te_norm, y_test, b_indices)
        info_metrics = StatisticalProofEngine.compute_information_theoretic_metrics(X_te_norm, y_test, b_indices)
        row = {
            "Configuration": name,
            "Band_Count": len(b_indices),
            "Fisher_Scatter_J": J_fisher,
            "Condition_Number_kappa": info_metrics["condition_number"],
            "Mean_Mutual_Information": info_metrics["mean_mi_with_identity"],
            "Mean_Interband_Correlation": info_metrics["mean_interband_correlation"],
        }
        info_results.append(row)
        print(f"    Fisher J: {J_fisher:.4f} | Condition Number: {info_metrics['condition_number']:.2f} | Mean MI: {info_metrics['mean_mi_with_identity']:.4f} | Interband Corr: {info_metrics['mean_interband_correlation']:.4f}")

    info_csv_path = "outputs/tables/information_theoretic_proofs.csv"
    with open(info_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(info_results[0].keys()))
        writer.writeheader()
        writer.writerows(info_results)
    print(f"  Saved information-theoretic proofs -> {info_csv_path}")

    # 7. MULTI-CLASSIFIER GENERALIZATION BENCHMARK
    print("\n" + "=" * 80)
    print("PHASE C: CROSS-CLASSIFIER GENERALIZATION BENCHMARK")
    print("=" * 80)

    benchmarker = MultiClassifierBenchmarker(random_seed=42)
    classifier_comparison = {}
    classifier_rows = []

    for name, b_indices in subsets.items():
        print(f"  Benchmarking 4 classifiers on: {name}...")
        clf_res = benchmarker.evaluate_all_classifiers(
            X_tr_norm[:, b_indices],
            y_train,
            X_te_norm[:, b_indices],
            y_test,
        )
        classifier_comparison[name] = {clf: res["accuracy"] for clf, res in clf_res.items()}
        for clf, res in clf_res.items():
            print(f"    {clf:16s} -> Accuracy: {res['accuracy']*100:.2f}%, F1: {res['f1_macro']:.4f}, Kappa: {res['kappa']:.4f}")
            classifier_rows.append({
                "Configuration": name,
                "Classifier": clf,
                "Accuracy": res["accuracy"],
                "F1_Macro": res["f1_macro"],
                "Kappa": res["kappa"],
            })

    clf_csv_path = "outputs/tables/multi_classifier_comparison.csv"
    with open(clf_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(classifier_rows[0].keys()))
        writer.writeheader()
        writer.writerows(classifier_rows)
    print(f"  Saved multi-classifier comparison -> {clf_csv_path}")

    # 8. 2D MANIFOLD CLUSTER SEPARABILITY (t-SNE & PCA)
    print("\n" + "=" * 80)
    print("PHASE D: 2D MANIFOLD PROJECTIONS & CLUSTER SEPARABILITY")
    print("=" * 80)

    manifold_eval = ManifoldClusterEvaluator(random_seed=42)
    embeddings_dict = {}
    manifold_rows = []

    for name, b_indices in subsets.items():
        print(f"  Computing manifold embeddings for {name}...")
        emb = manifold_eval.evaluate_embedding(X_te_norm, y_test, bands=b_indices, n_samples=1600)
        embeddings_dict[name] = emb
        row = {
            "Configuration": name,
            "Silhouette_Score": emb["silhouette_score"],
            "Davies_Bouldin_Index": emb["davies_bouldin_index"],
            "Calinski_Harabasz_Score": emb["calinski_harabasz_score"],
            "PCA_Explained_Variance_2D": emb["pca_explained_variance"],
        }
        manifold_rows.append(row)
        print(f"    Silhouette: {emb['silhouette_score']:.4f} | Davies-Bouldin: {emb['davies_bouldin_index']:.4f} | Calinski-Harabasz: {emb['calinski_harabasz_score']:.2f}")

    manifold_csv_path = "outputs/tables/manifold_separability_metrics.csv"
    with open(manifold_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifold_rows)
    print(f"  Saved manifold metrics -> {manifold_csv_path}")

    # 9. GENERATE PUBLICATION FIGURES
    print("\n" + "=" * 80)
    print("PHASE E: GENERATING ADVANCED SCIENTIFIC FIGURES")
    print("=" * 80)

    vis = AdvancedVisualizer(output_dir="outputs/figures")

    # Figure 1: 4-Panel t-SNE Cluster Separability
    p1 = vis.plot_manifold_separability_comparison(
        embeddings_dict,
        subject_names,
        filename="tsne_cluster_separability.png",
    )
    print(f"  [1/4] Generated t-SNE separability plot -> {p1}")

    # Figure 2: Normalized Confusion Matrix for PO-3
    clf_nc = FastNearestCentroid(metric="cosine").fit(X_tr_norm[:, best_3_filtered], y_train)
    y_pred_po3 = clf_nc.predict(X_te_norm[:, best_3_filtered])
    p2 = vis.plot_confusion_matrix(
        y_test,
        y_pred_po3,
        subject_names,
        filename="confusion_matrix_po3.png",
    )
    print(f"  [2/4] Generated confusion matrix -> {p2}")

    # Figure 3: Multi-Criteria Algorithmic Superiority Radar Chart
    categories = [
        "Fitness Quality\n(1 - Fit)",
        "Test Accuracy\n(OA)",
        "Macro F1-Score\n(F1)",
        "Low Redundancy\n(1 - Red)",
        "Search Speed\n(1 / Runtime)",
        "Stability\n(1 - Std)",
    ]

    radar_data = {}
    algos = ["PUMA", "PSO", "WOA", "GWO", "DE"]

    # Gather metrics across runs
    raw_fit = {a: np.mean([r["best_fitness"] for r in comp_metrics[a]]) for a in algos}
    raw_acc = {a: np.mean([r["test_accuracy"] for r in comp_metrics[a]]) for a in algos}
    raw_f1 = {a: np.mean([r["test_f1"] for r in comp_metrics[a]]) for a in algos}
    raw_red = {a: np.mean([r["redundancy"] for r in comp_metrics[a]]) for a in algos}
    raw_time = {a: np.mean([r["runtime_sec"] for r in comp_metrics[a]]) for a in algos}
    raw_std = {a: np.std([r["best_fitness"] for r in comp_metrics[a]]) for a in algos}

    def norm_dim(val_dict, invert=False):
        vals = np.array(list(val_dict.values()))
        if invert:
            vals = -vals
        min_v, max_v = np.min(vals), np.max(vals)
        if max_v == min_v:
            return {a: 0.8 for a in val_dict}
        return {a: float(0.35 + 0.65 * (vals[i] - min_v) / (max_v - min_v)) for i, a in enumerate(val_dict.keys())}

    n_fit = norm_dim(raw_fit, invert=True)
    n_acc = norm_dim(raw_acc, invert=False)
    n_f1 = norm_dim(raw_f1, invert=False)
    n_red = norm_dim(raw_red, invert=True)
    n_speed = norm_dim(raw_time, invert=True)
    n_stab = norm_dim(raw_std, invert=True)

    for a in algos:
        radar_data[a] = [
            n_fit[a],
            n_acc[a],
            n_f1[a],
            n_red[a],
            n_speed[a],
            n_stab[a],
        ]

    p3 = vis.plot_radar_chart(
        radar_data,
        categories,
        filename="radar_algorithm_comparison.png",
    )
    print(f"  [3/4] Generated radar comparison -> {p3}")

    # Figure 4: Cross-Classifier Generalization Bar Chart
    p4 = vis.plot_multi_classifier_generalization(
        classifier_comparison,
        filename="multiclassifier_generalization.png",
    )
    print(f"  [4/4] Generated cross-classifier comparison -> {p4}")

    print("\n" + "=" * 80)
    print("ADVANCED PROOFS EXECUTION COMPLETE! ALL TABLES & FIGURES GENERATED.")
    print("=" * 80)


if __name__ == "__main__":
    main()
