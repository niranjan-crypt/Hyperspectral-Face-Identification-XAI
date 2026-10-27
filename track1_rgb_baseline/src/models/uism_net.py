"""
Unified Identity-Spectral Manifold Network (UISM-Net) Implementation.
Combines an ImageNet-pretrained frozen MobileNetV2 backbone with a 128-dim
L2-normalized unit hypersphere embedding and an identity softmax classification head.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_uism_net(
    input_shape=(128, 128, 3),
    embedding_dim=128,
    num_classes=7,
    dropout_rate=0.30,
    freeze_backbone=True,
):
    """
    Constructs full UISM-Net for end-to-end training.
    """
    # 1. Data Augmentation
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="uism_augmentation",
    )

    # 2. Pretrained MobileNetV2 Feature Extractor
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    if freeze_backbone:
        base.trainable = False

    # 3. Model Flow
    inputs = layers.Input(shape=input_shape, name="input_image")
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    # 4. Manifold Projection & L2 Normalization onto Unit Hypersphere S^(d-1)
    raw_embedding = layers.Dense(embedding_dim, name="uism_embedding")(x)
    normalized_embedding = layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name="unit_hypersphere_embedding",
    )(raw_embedding)

    # 5. Classification Head (Active during training, discarded during metric matching)
    outputs = layers.Dense(
        num_classes, activation="softmax", name="classification_head"
    )(normalized_embedding)

    model = models.Model(inputs, outputs, name="UISM_Net_Full")
    return model


def build_uism_embedding_extractor(full_model):
    """
    Extracts the normalized hyperspherical embedding sub-network for inference/matching.
    """
    return models.Model(
        inputs=full_model.get_layer("input_image").input,
        outputs=full_model.get_layer("unit_hypersphere_embedding").output,
        name="UISM_Net_EmbeddingExtractor",
    )
