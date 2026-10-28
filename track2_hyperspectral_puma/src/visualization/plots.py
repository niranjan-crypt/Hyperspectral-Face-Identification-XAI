"""
Publication-quality scientific visualization module.
Generates convergence curves, band optimization metric boxplots, spectral band selection histograms,
mean spectral signatures of human subjects, and inter-band correlation heatmaps.
"""

import os
from typing import Dict, List, Optional, Sequence, Any
import numpy as np

# Ensure matplotlib uses a safe local cache directory
os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs/cache/matplotlib")
import matplotlib.pyplot as plt


class Visualizer:
    """Creates scientific plots and figures for band optimization reporting."""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Use clean scientific style
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14,
            "figure.dpi": 300,
        })

    def plot_convergence(self, benchmark_results: Dict[str, List[Dict[str, Any]]], filename: str = "convergence_comparison.png") -> str:
        """Plots mean convergence curves with ±1 std deviation shaded error bands."""
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"PUMA": "#D95F02", "PSO": "#7570B3", "WOA": "#1B9E77", "GWO": "#E7298A", "DE": "#66A61E"}

        for algo, runs in benchmark_results.items():
            curves = np.array([r["convergence"] for r in runs])
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            iters = np.arange(len(mean_curve))

            color = colors.get(algo, None)
            ax.plot(iters, mean_curve, label=algo, lw=2.0, color=color)
            ax.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15, color=color)

        ax.set_title("Band Optimization Convergence (Mean ± 1 Std across Runs)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Fitness Score (Minimization)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_metrics_comparison(self, benchmark_results: Dict[str, List[Dict[str, Any]]], filename: str = "metrics_boxplot.png") -> str:
        """Plots boxplots for Fitness Score, Inter-Band Redundancy, and Information Entropy."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        algos = list(benchmark_results.keys())

        fitness_data = [[r["best_fitness"] for r in benchmark_results[a]] for a in algos]
        redundancy_data = [[r["redundancy"] for r in benchmark_results[a]] for a in algos]
        entropy_data = [[r["information_entropy"] for r in benchmark_results[a]] for a in algos]

        # 1. Best Fitness
        bp1 = axes[0].boxplot(fitness_data, labels=algos, patch_artist=True)
        axes[0].set_title("Optimization Fitness (Lower is Better)")
        axes[0].set_ylabel("Fitness Value")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # 2. Inter-Band Redundancy
        bp2 = axes[1].boxplot(redundancy_data, labels=algos, patch_artist=True)
        axes[1].set_title("Inter-Band Redundancy (Lower is Better)")
        axes[1].set_ylabel("Mean Pairwise Pearson Correlation")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # 3. Information Entropy
        bp3 = axes[2].boxplot(entropy_data, labels=algos, patch_artist=True)
        axes[2].set_title("Information Entropy (Higher is Better)")
        axes[2].set_ylabel("Shannon Entropy (Normalized)")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        palette = ["#D95F02", "#7570B3", "#1B9E77", "#E7298A", "#66A61E"]
        for bp in [bp1, bp2, bp3]:
            for patch, color in zip(bp["boxes"], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_band_distribution(self, benchmark_results: Dict[str, List[Dict[str, Any]]], filename: str = "band_distribution_histogram.png") -> str:
        """Plots histogram showing selected wavelength distributions across algorithms."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"PUMA": "#D95F02", "PSO": "#7570B3", "WOA": "#1B9E77", "GWO": "#E7298A", "DE": "#66A61E"}

        bins = np.linspace(390, 1020, 45)
        for algo, runs in benchmark_results.items():
            all_wls = [wl for r in runs for wl in r["wavelengths"]]
            ax.hist(all_wls, bins=bins, alpha=0.5, label=algo, color=colors.get(algo, None), edgecolor="black", linewidth=0.5)

        ax.set_title("Distribution of Selected Spectral Channels across Metaheuristics")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Selection Frequency across Runs")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_spectral_signatures(
        self,
        wavelengths: np.ndarray,
        X_train: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        subject_names: Optional[List[str]] = None,
        selected_3_wavelengths: Optional[Sequence[float]] = None,
        filename: str = "spectral_signatures.png",
    ) -> str:
        """Plots mean reflectance spectral signatures with selected optimal bands marked."""
        fig, ax = plt.subplots(figsize=(10, 5.5))

        if y_train is not None and subject_names is not None:
            classes = np.unique(y_train)
            for c in classes:
                mask = (y_train == c)
                mean_sig = np.mean(X_train[mask], axis=0)
                std_sig = np.std(X_train[mask], axis=0)
                name = subject_names[c] if c < len(subject_names) else f"Subject {c+1}"
                p = ax.plot(wavelengths, mean_sig, lw=1.8, label=name)
                ax.fill_between(wavelengths, mean_sig - 0.5 * std_sig, mean_sig + 0.5 * std_sig, alpha=0.1, color=p[0].get_color())
        else:
            mean_sig = np.mean(X_train, axis=0)
            std_sig = np.std(X_train, axis=0)
            ax.plot(wavelengths, mean_sig, lw=2.0, color="#1A365D", label="Mean Spectral Profile")
            ax.fill_between(wavelengths, mean_sig - std_sig, mean_sig + std_sig, alpha=0.15, color="#2B6CB0")

        if selected_3_wavelengths is not None:
            for idx, wl in enumerate(selected_3_wavelengths):
                ax.axvline(
                    wl,
                    color="#D95F02",
                    linestyle="--",
                    lw=2.2,
                    alpha=0.9,
                    label=f"Optimal Band {idx+1} ({wl:.1f} nm)",
                )

        ax.set_title("Calibrated Facial Spectral Reflectance Signatures with PUMA Optimal Bands")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance (Normalized)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", ncol=2, frameon=True)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_correlation_heatmap(
        self,
        full_corr: np.ndarray,
        selected_bands: Sequence[int],
        selected_wavelengths: Sequence[float],
        filename: str = "correlation_heatmap.png",
    ) -> str:
        """Plots comparison heatmaps: full spectrum correlation vs. optimized subset correlation."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. Full Spectrum Correlation (Subsampled for clear viewing)
        sub_sample = max(1, len(full_corr) // 100)
        im1 = axes[0].imshow(full_corr[::sub_sample, ::sub_sample], cmap="viridis", vmin=0, vmax=1)
        axes[0].set_title(f"Full Spectrum Correlation ({len(full_corr)} Bands)\nHigh Inter-Band Redundancy")
        axes[0].set_xlabel("Band Index")
        axes[0].set_ylabel("Band Index")
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        # 2. Optimized Subset Correlation
        sub_corr = full_corr[np.ix_(selected_bands, selected_bands)]
        im2 = axes[1].imshow(sub_corr, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title(f"Optimized Band Subset Correlation ({len(selected_bands)} Bands)\nMinimal Redundancy Decoupled")
        axes[1].set_xlabel("Selected Band Index")
        axes[1].set_ylabel("Selected Band Index")
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
