"""
Grad-CAM and Grad-CAM++ implementation for facial feature saliency and visual interpretability.
Implements Equations 3.60 and 3.61 from the Capstone Report:
alpha_k^c = (1/Z) * sum_i sum_j (d y^c / d A_ij^k)   [Eq. 3.60]
L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k)             [Eq. 3.61]
"""

import cv2
import numpy as np
import tensorflow as tf


class GradCAMExplainer:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        if target_layer_name is None:
            # Pick last convolutional layer
            for layer in reversed(model.layers):
                if "conv" in layer.name.lower():
                    target_layer_name = layer.name
                    break
        self.target_layer_name = target_layer_name
        self.grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(self.target_layer_name).output, model.output],
        )

    def generate_heatmap(self, img_tensor, class_idx=None):
        """
        img_tensor: shape (1, H, W, C)
        Returns:
            heatmap: 2D numpy array normalized in [0, 1]
        """
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # Gradients of target class score w.r.t. feature map activations (Eq. 3.60)
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients: alpha_k^c
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Linear combination: L_GradCAM^c = ReLU(sum_k alpha_k * A^k) (Eq. 3.61)
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0.0)

        # Normalize heatmap to [0, 1]
        max_val = tf.reduce_max(heatmap)
        if max_val > 0:
            heatmap /= max_val

        return heatmap.numpy()

    def overlay_heatmap(self, orig_img_rgb, heatmap, alpha=0.40, colormap=cv2.COLORMAP_JET):
        """
        Overlays Grad-CAM heatmap on top of original RGB image.
        """
        h, w = orig_img_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB)

        overlay = (alpha * colored_heatmap + (1.0 - alpha) * orig_img_rgb).astype(np.uint8)
        return overlay, colored_heatmap
