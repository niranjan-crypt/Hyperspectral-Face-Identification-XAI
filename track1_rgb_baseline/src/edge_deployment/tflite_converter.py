"""
Model Quantization and TFLite Conversion for Edge Deployment on Raspberry Pi 4.
Supports Default (Float32), Float16, and Full INT8 Quantization.
"""

import os
import tensorflow as tf


def convert_to_tflite(
    keras_model_or_path,
    output_tflite_path,
    quantization="float16",
    representative_dataset_gen=None,
):
    """
    quantization: "none" (FP32), "float16" (FP16), or "int8" (Full INT8)
    """
    if isinstance(keras_model_or_path, str):
        model = tf.keras.models.load_model(keras_model_or_path)
    else:
        model = keras_model_or_path

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_dataset_gen is not None:
            converter.representative_dataset = representative_dataset_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    with open(output_tflite_path, "wb") as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(output_tflite_path) / (1024 * 1024)
    print(f"Exported TFLite model to {output_tflite_path} ({size_mb:.2f} MB)")
    return output_tflite_path
