"""
Accuracy Progression and Whole-Face Identification Visualization Module.
Plots:
    1. Progression Bar Chart: Isolated Pixels -> Micro-Patches -> Regional Blocks -> Whole-Face Decisions
    2. Whole-Face ROI Identification Confusion Matrices (PO-3 and PUMA 30 Bands)
"""

import os
from typing import Dict, List, Any
import numpy as np

os.environ["MPLCONFIGDIR"] = os.path.abspath("outputs/cache/matplotlib")
import matplotlib.pyplot as plt


class AccuracyVisualizer:
    """Visualizes the architectural elevation of face identification accuracy."""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        plt.rcParams.update({
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "figure.dpi": 300,
        })

    def plot_accuracy_progression(
        self,
        progression_data: Dict[str, Dict[str, float]],
        filename: str = "accuracy_progression_bar.png",
    ) -> str:
        """
        Grouped bar chart showing accuracy climbing from isolated pixels
        to regional patches, and finally to whole-face biometric consensus.
        """
        configurations = list(progression_data.keys())
        stages = list(next(iter(progression_data.values())).keys())

        x = np.arange(len(stages))
        width = 0.8 / len(configurations)

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = ["#D95F02", "#1F78B4", "#2CA02C"]

        for idx, config_name in enumerate(configurations):
            accuracies = [progression_data[config_name][stage] * 100 for stage in stages]
            offset = (idx - len(configurations) / 2 + 0.5) * width
            rects = ax.bar(
                x + offset,
                accuracies,
                width,
                label=config_name,
                color=colors[idx % len(colors)],
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
            )

            for rect in rects:
                height = rect.get_height()
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold" if height >= 70.0 else "normal",
                )

        ax.set_ylabel("Identification Accuracy (%)", fontweight="bold")
        ax.set_title("Accuracy Elevation Progression: Microscopic Pixels to Whole-Face Decisions", pad=15, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(stages, fontweight="bold", fontsize=10)
        ax.set_ylim(0, 115)
        ax.axhline(100.0, color="gray", linestyle=":", alpha=0.6)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        ax.legend(loc="upper left", frameon=True, fontsize=10)

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path

    def plot_whole_face_confusion_matrices(
        self,
        cm_po3: np.ndarray,
        cm_puma30: np.ndarray,
        subject_names: List[str],
        filename: str = "whole_face_confusion_matrix.png",
    ) -> str:
        """Plots side-by-side whole-face decision confusion matrices for PO-3 vs PUMA 30 Bands."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        n_classes = len(subject_names)
        short_names = [f"Sub {i+1}" for i in range(n_classes)]

        acc_po3 = float(np.trace(cm_po3)) / float(max(1, np.sum(cm_po3))) * 100.0
        acc_puma30 = float(np.trace(cm_puma30)) / float(max(1, np.sum(cm_puma30))) * 100.0
        hits_po3 = int(np.trace(cm_po3))
        hits_puma30 = int(np.trace(cm_puma30))

        matrices = [
            (axes[0], cm_po3, f"PO-3 Optimal Triad (3 Bands)\nWhole-Face Accuracy: {acc_po3:.1f}% ({hits_po3}/{n_classes} Faces Correct)", plt.cm.Blues),
            (axes[1], cm_puma30, f"PUMA Best 30-Band Subset\nWhole-Face Accuracy: {acc_puma30:.1f}% ({hits_puma30}/{n_classes} Faces Correct)", plt.cm.Greens),
        ]

        for ax, cm, title, cmap in matrices:
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, pad=12, fontweight="bold")
            tick_marks = np.arange(n_classes)
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
            ax.set_yticklabels(short_names, fontsize=9)
            ax.set_xlabel("Predicted Subject")
            ax.set_ylabel("True Subject")

            for i in range(n_classes):
                for j in range(n_classes):
                    val = cm[i, j]
                    is_diag = (i == j)
                    color = "white" if val > 0.5 else "black"
                    txt = "1 (HIT)" if val == 1 and is_diag else ("0" if val == 0 else str(val))
                    ax.text(
                        j, i, txt,
                        ha="center", va="center",
                        color=color,
                        fontweight="bold" if is_diag and val > 0 else "normal",
                        fontsize=10,
                    )

        save_path = os.path.join(self.output_dir, filename)
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        return save_path
