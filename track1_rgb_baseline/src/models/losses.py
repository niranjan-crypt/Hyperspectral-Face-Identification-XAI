"""
Loss formulations for Face Identification:
- Contrastive metric learning loss (Eq. 3.50)
- Hybrid cross-entropy + metric loss (Eq. 3.51)
- ArcFace (Additive Angular Margin) & CosFace (Large Margin Cosine) losses
"""

import math
import tensorflow as tf


class ContrastiveMetricLoss(tf.keras.losses.Loss):
    """
    Contrastive Metric Loss:
    L_metric = y_ij * d_ij^2 + (1 - y_ij) * [max(0, m - d_ij)]^2   [Equation 3.50]
    """
    def __init__(self, margin=1.0, name="contrastive_metric_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, d_ij):
        y_true = tf.cast(y_true, tf.float32)
        pos_loss = y_true * tf.square(d_ij)
        neg_loss = (1.0 - y_true) * tf.square(tf.maximum(self.margin - d_ij, 0.0))
        return tf.reduce_mean(pos_loss + neg_loss)


class ArcFaceLoss(tf.keras.losses.Loss):
    """
    ArcFace: Additive Angular Margin Loss over normalized embeddings.
    Adds angular margin m to target class logit: cos(theta + m)
    """
    def __init__(self, scale=30.0, margin=0.50, name="arcface_loss"):
        super().__init__(name=name)
        self.scale = scale
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def call(self, y_true, cosine):
        # cosine: (batch_size, num_classes)
        sine = tf.sqrt(1.0 - tf.square(cosine) + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = tf.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(cosine)[-1])
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, output, from_logits=True)
