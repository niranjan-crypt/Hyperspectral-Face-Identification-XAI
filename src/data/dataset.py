"""
Dataset loading utilities for RGB face identification and hyperspectral cube batches.
"""

import os
import cv2
import numpy as np
import tensorflow as tf


def load_raw_dataset(folder_path, img_size=(128, 128)):
    """
    Loads raw images and numeric class labels as NumPy arrays.
    Returns:
        X (np.ndarray): Shape (N, H, W, 3)
        y (np.ndarray): Shape (N,)
        class_names (list): Sorted list of class directory names
    """
    class_names = sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))])
    X, y = [], []

    for idx, cls in enumerate(class_names):
        cls_dir = os.path.join(folder_path, cls)
        for fname in os.listdir(cls_dir):
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            if img_size is not None:
                img = cv2.resize(img, img_size)
            X.append(img)
            y.append(idx)

    return np.array(X), np.array(y), class_names


def create_tf_dataset(folder_path, img_size=(128, 128), batch_size=16, shuffle=True, seed=42):
    """
    Creates an optimized tf.data.Dataset pipeline with prefetching.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        folder_path,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )
    autotune = tf.data.AUTOTUNE
    return ds.prefetch(autotune)
