"""
Standalone script to train UISM-Net and evaluate against comparative baselines.
Usage:
    python train_rgb_baseline.py --data_dir data/processed/rgb/dataset_split --epochs 40
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.models.uism_net import build_uism_net
from src.models.baselines import (
    train_lbp_svm,
    train_hog_svm,
    build_transfer_baseline,
    extract_lbp_features,
    extract_hog_features,
)
from src.data.dataset import load_raw_dataset, create_tf_dataset
from src.utils.metrics import compute_classification_metrics
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_model_comparison,
    plot_multi_confusion_matrices,
)


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark RGB face identification models.")
    parser.add_argument("--config", type=str, default="configs/rgb_baseline.yaml", help="Path to config YAML")
    parser.add_argument("--data_dir", type=str, default=None, help="Root dataset directory")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save plots and reports")
    args = parser.parse_args()

    # Load config if present
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    split_dir = args.data_dir or cfg.get("dataset", {}).get("split_dir", "data/processed/rgb/dataset_split")
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")
    test_dir = os.path.join(split_dir, "test")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    print(f"Loading data from {split_dir}...")
    train_ds = create_tf_dataset(train_dir, img_size=(128, 128), batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_dir, img_size=(128, 128), batch_size=args.batch_size, shuffle=False)
    test_ds = create_tf_dataset(test_dir, img_size=(128, 128), batch_size=args.batch_size, shuffle=False)

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Compute class weights
    train_labels = np.concatenate([y.numpy() for _, y in train_ds])
    unique_classes = np.unique(train_labels)
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # --- Train UISM-Net ---
    print("\n--- Training Proposed UISM-Net ---")
    uism_model = build_uism_net(num_classes=num_classes)
    uism_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6),
    ]

    history = uism_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight_dict,
    )

    model_save_path = "saved_models/rgb_face_cnn_model.keras"
    uism_model.save(model_save_path)
    print(f"Saved trained UISM-Net to {model_save_path}")

    # Evaluate UISM-Net
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = uism_model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    uism_metrics = compute_classification_metrics(y_true, y_pred)
    print(f"\nUISM-Net Test Results: {uism_metrics}")

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=f"{args.output_dir}/confusion_matrix.png")
    plot_training_curves(history.history, save_path=f"{args.output_dir}/training_curves.png")

    print(f"\nTraining pipeline completed. Results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
