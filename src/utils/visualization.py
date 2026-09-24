"""
Visualization utilities for confusion matrices, training curves, and model comparisons.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, class_names, title="UISM-Net Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gcf()


def plot_training_curves(history_dict, title="Training vs Validation Dynamics", save_path=None):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    if "accuracy" in history_dict:
        plt.plot(history_dict["accuracy"], label="Train Acc")
    if "val_accuracy" in history_dict:
        plt.plot(history_dict["val_accuracy"], label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    if "loss" in history_dict:
        plt.plot(history_dict["loss"], label="Train Loss")
    if "val_loss" in history_dict:
        plt.plot(history_dict["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gcf()


def plot_model_comparison(results_df, title="Model Comparison — RGB Face Identification", save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.plot(kind="bar", ax=ax, title=title)
    ax.set_ylabel("Score")
    ax.set_ylim([0.0, 1.05])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_multi_confusion_matrices(preds_store, class_names, save_path=None):
    n_models = len(preds_store)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for ax, (name, (yt, yp)) in zip(axes_flat, preds_store.items()):
        cm = confusion_matrix(yt, yp)
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
