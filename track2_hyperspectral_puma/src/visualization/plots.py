"""
Publication-quality scientific visualization module.
Generates convergence curves, metric boxplots, spectral band selection histograms,
mean spectral signatures of the 7 subjects, and 3-band composite visualizations.
"""

import os
from typing import Dict, List, Optional, Sequence, Any
import numpy as np

# Ensure matplotlib uses a safe local cache directory
os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs/cache/matplotlib")
import matplotlib.pyplot as plt


class Visualizer:
    """Creates scientific plots and figures for experimental reporting."""

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

        ax.set_title("Optimization Convergence Curves (Mean ± 1 Std across Runs)")
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
        """Plots boxplots for Overall Accuracy, F1-macro, and Cohen's Kappa."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        algos = list(benchmark_results.keys())

        oa_data = [[r["test_accuracy"] * 100 for r in benchmark_results[a]] for a in algos]
        f1_data = [[r["test_f1"] for r in benchmark_results[a]] for a in algos]
        kappa_data = [[r["test_kappa"] for r in benchmark_results[a]] for a in algos]

        # Overall Accuracy
        bp1 = axes[0].boxplot(oa_data, labels=algos, patch_artist=True)
        axes[0].set_title("Overall Accuracy (%)")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # F1 Macro
        bp2 = axes[1].boxplot(f1_data, labels=algos, patch_artist=True)
        axes[1].set_title("Macro F1-Score")
        axes[1].set_ylabel("F1 Score")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        # Kappa
        bp3 = axes[2].boxplot(kappa_data, labels=algos, patch_artist=True)
        axes[2].set_title("Cohen's Kappa Coefficient")
        axes[2].set_ylabel("Kappa")
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
        y_train: np.ndarray,
        subject_names: List[str],
        selected_3_wavelengths: Optional[Sequence[float]] = None,
        filename: str = "spectral_signatures.png",
    ) -> str:
        """Plots mean reflectance spectral signature for each of the 7 subjects with selected 3 bands marked."""
        fig, ax = plt.subplots(figsize=(10, 5.5))
        classes = np.unique(y_train)

        for c in classes:
            mask = (y_train == c)
            mean_sig = np.mean(X_train[mask], axis=0)
            std_sig = np.std(X_train[mask], axis=0)
            name = subject_names[c] if c < len(subject_names) else f"Subject {c+1}"
            p = ax.plot(wavelengths, mean_sig, lw=1.8, label=name)
            ax.fill_between(wavelengths, mean_sig - 0.5 * std_sig, mean_sig + 0.5 * std_sig, alpha=0.1, color=p[0].get_color())

        if selected_3_wavelengths is not None:
            for idx, wl in enumerate(selected_3_wavelengths):
                ax.axvline(
                    wl,
                    color="#D95F02",
                    linestyle="--",
                    lw=2.0,
                    alpha=0.85,
                    label=f"Selected Band {idx+1} ({wl:.1f} nm)" if idx == 0 else f"Band {idx+1} ({wl:.1f} nm)",
                )

        ax.set_title("Calibrated Spectral Signatures of 7 Human Subjects with Selected 3 Bands")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Reflectance (Normalized)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", ncol=2, frameon=True)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
