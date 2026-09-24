"""
Spectral-Spatial 3D-CNN for Reduced-Band Hyperspectral Face Identification.
Implements Equations 3.43 to 3.47 from the Capstone Report:
Joint 3D convolution across spatial (H, W) and spectral (K) dimensions.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_spectral_spatial_3dcnn(
    input_shape=(128, 128, 10, 1),
    embedding_dim=128,
    num_classes=7,
    dropout_rate=0.30,
):
    """
    input_shape: (H, W, K, Channels=1) where K is number of metaheuristic-selected bands.
    """
    inputs = layers.Input(shape=input_shape, name="hsi_cube_input")

    # Block 1: 3D Spectral-Spatial Convolution (Eq. 3.44)
    x = layers.Conv3D(16, kernel_size=(3, 3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 1))(x)

    # Block 2
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Block 3
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling3D()(x)

    # Dense Projection to Latent Identity Embedding (Eq. 3.45)
    x = layers.Dropout(dropout_rate)(x)
    raw_embedding = layers.Dense(embedding_dim, name="hsi_embedding")(x)

    # L2 Normalization onto Unit Hypersphere (Eq. 3.46)
    normalized_embedding = layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="unit_hypersphere_embedding",
    )(raw_embedding)

    # Softmax Classification Head (Eq. 3.47)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="classification_head"
    )(normalized_embedding)

    model = models.Model(inputs, outputs, name="SpectralSpatial_3DCNN")
    return model
