"""
Baseline and Comparative Architectures for Face Identification Benchmarking:
- Handcrafted features: LBP + RBF SVM, HOG + RBF SVM
- Deep transfer baselines: ResNet-50, DenseNet-121, EfficientNetB0
[Report Section 4.2]
"""

import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import local_binary_pattern, hog
import tensorflow as tf
from tensorflow.keras import layers, models


# --- Handcrafted Baselines ---
def extract_lbp_features(images, P=8, R=1):
    feats = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float32") / (hist.sum() + 1e-6)
        feats.append(hist)
    return np.array(feats)


def train_lbp_svm(X_train, y_train, C=10.0):
    feats_train = extract_lbp_features(X_train)
    clf = SVC(kernel="rbf", C=C, gamma="scale", probability=True)
    clf.fit(feats_train, y_train)
    return clf


def extract_hog_features(images):
    feats = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
        )
        feats.append(feat)
    return np.array(feats)


def train_hog_svm(X_train, y_train, C=10.0):
    feats_train = extract_hog_features(X_train)
    clf = SVC(kernel="rbf", C=C, gamma="scale", probability=True)
    clf.fit(feats_train, y_train)
    return clf


# --- Deep Transfer Baselines ---
def build_transfer_baseline(backbone_name, num_classes=7, input_shape=(128, 128, 3), dropout=0.3):
    backbone_dict = {
        "ResNet-50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet50.preprocess_input),
        "DenseNet-121": (tf.keras.applications.DenseNet121, tf.keras.applications.densenet.preprocess_input),
        "EfficientNetB0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
    }

    if backbone_name not in backbone_dict:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(backbone_dict.keys())}")

    backbone_fn, preprocess_fn = backbone_dict[backbone_name]

    base = backbone_fn(input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = preprocess_fn(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name=f"Baseline_{backbone_name}")
    return model
