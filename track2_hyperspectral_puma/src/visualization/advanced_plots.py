"""
Advanced Scientific Visualization Suite for Hyperspectral Face Identification.
Generates:
    1. 4-Panel 2D Manifold Cluster Projections (t-SNE / PCA)
    2. Normalized 7x7 Confusion Matrix Heatmap
    3. Multi-Attribute Radar / Spider Benchmark Chart
    4. Cross-Classifier Generalization Grouped Bar Chart
"""

import os
from typing import Dict, List, Optional, Sequence, Any
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs/cache/matplotlib")
import matplotlib.pyplot as plt


class AdvancedVisualizer:
    """Produces publication-grade figures for statistical and manifold proofs."""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "figure.dpi": 300,
        })

    def plot_manifold_separability_comparison(
        self,
        embeddings_dict: Dict[str, Dict[str, Any]],
        subject_names: List[str],
        filename: str = "tsne_cluster_separability.png",
    ) -> str:
        """
        Plots a 2x2 grid comparing 2D t-SNE cluster embeddings for:
            - PO-3 (PUMA Optimal 3 Bands)
            - RGB-3 (Standard Visible Spectrum)
            - Full Spectrum (925 Contiguous Bands)
            - Random 3 Bands
        Annotates Silhouette Score (higher is better) and Davies-Bouldin Index (lower is better).
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        ax_flat = axes.flatten()

        subsets = list(embeddings_dict.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(subject_names)))

        for i, subset_name in enumerate(subsets):
            ax = ax_flat[i]
            data = embeddings_dict[subset_name]
            coords = data["tsne_coords"]
            y = data["y"]
            sil = data["silhouette_score"]
            db = data["davies_bouldin_index"]

            for c_idx, name in enumerate(subject_names):
                mask = (y == c_idx)
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    color=colors[c_idx],
                    label=name if i == 0 else None,
                    alpha=0.65,
                    s=18,
                    edgecolors="none",
                )

            ax.set_title(
                f"{subset_name}\n"
                f"Silhouette: {sil:.4f} (Higher=Better) | Davies-Bouldin: {db:.3f} (Lower=Better)",
                fontweight="bold" if "PUMA" in subset_name or "PO-3" in subset_name else "normal",
            )
            ax.set_xlabel("t-SNE Dimension 1")
            ax.set_ylabel("t-SNE Dimension 2")
            ax.grid(True, linestyle=":", alpha=0.4)

        # Global legend from first subplot
        handles, labels = ax_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(subject_names),
            frameon=True,
            bbox_to_anchor=(0.5, 0.02),
        )

        fig.suptitle(
            "t-SNE Manifold Cluster Separability of 7 Human Subjects Across Spectral Subsets",
            fontsize=15,
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])

        save_path = os.path.join(self.output_dir, filename)
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        subject_names: List[str],
        filename: str = "confusion_matrix_po3.png",
    ) -> str:
        """Plots a normalized confusion matrix heatmap for the PO-3 test predictions."""
        n_classes = len(subject_names)
        cm_counts = np.zeros((n_classes, n_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm_counts[int(t), int(p)] += 1

        # Normalize by true class row sums
        row_sums = cm_counts.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_counts, row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Classification Probability / Recall", rotation=270, labelpad=15)

        tick_marks = np.arange(n_classes)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(subject_names, rotation=35, ha="right")
        ax.set_yticklabels(subject_names)

        thresh = cm_norm.max() / 2.0
        for i in range(n_classes):
            for j in range(n_classes):
                pct = cm_norm[i, j] * 100
                cnt = cm_counts[i, j]
                ax.text(
                    j,
                    i,
                    f"{pct:.1f}%\n({cnt})",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontsize=9,
                )

        ax.set_ylabel("True Subject Identity")
        ax.set_xlabel("Predicted Subject Identity")
        ax.set_title("Normalized Confusion Matrix (Held-out Test Set, PO-3 Bands)", pad=15)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_radar_chart(
        self,
        radar_metrics: Dict[str, List[float]],
        categories: List[str],
        filename: str = "radar_comparison.png",
    ) -> str:
        """
        Plots a radar/spider chart comparing the 5 metaheuristic algorithms across multiple dimensions.
        Metrics are normalized in [0, 1] (higher = superior).
        """
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete circle

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = {
            "PUMA": "#D95F02",
            "PSO": "#7570B3",
            "WOA": "#1B9E77",
            "GWO": "#E7298A",
            "DE": "#66A61E",
        }

        for algo, values in radar_metrics.items():
            val_circle = values + values[:1]
            c = colors.get(algo, "#333333")
            lw = 2.8 if algo == "PUMA" else 1.8
            ax.plot(angles, val_circle, color=c, linewidth=lw, label=algo)
            ax.fill(angles, val_circle, color=c, alpha=0.18 if algo == "PUMA" else 0.06)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)

        ax.set_title(
            "Multi-Criteria Algorithmic Superiority Radar Chart\n(Higher is Superior on All Axes)",
            size=14,
            pad=25,
            fontweight="bold",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=True)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_multi_classifier_generalization(
        self,
        classifier_data: Dict[str, Dict[str, float]],
        filename: str = "multiclassifier_generalization.png",
    ) -> str:
        """
        Grouped bar chart showing that PUMA's selected bands achieve superior accuracy across
        all 4 classifier paradigms: Nearest Centroid, Linear SVM, Random Forest, and k-NN.
        """
        classifiers = list(next(iter(classifier_data.values())).keys())
        subsets = list(classifier_data.keys())

        x = np.arange(len(classifiers))
        width = 0.8 / len(subsets)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        palette = ["#D95F02", "#1F78B4", "#33A02C", "#FB9A99"]

        for idx, subset in enumerate(subsets):
            accuracies = [classifier_data[subset][clf] * 100 for clf in classifiers]
            offset = (idx - len(subsets) / 2 + 0.5) * width
            rects = ax.bar(
                x + offset,
                accuracies,
                width,
                label=subset,
                color=palette[idx % len(palette)],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.5,
            )
            # Add value labels on top of bars
            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold" if "PO-3" in subset or "PUMA" in subset else "normal",
                )

        ax.set_ylabel("Identification Accuracy (%)")
        ax.set_title("Cross-Classifier Generalization: Accuracy across 4 Algorithmic Paradigms")
        ax.set_xticks(x)
        # Format names cleanly
        display_names = [c.replace("_", " ") for c in classifiers]
        ax.set_xticklabels(display_names, fontweight="bold")
        ax.set_ylim(0, max([max(classifier_data[s].values()) for s in subsets]) * 100 * 1.25)
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")
        ax.legend(loc="upper left", frameon=True)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
