"""
Edge Performance Metrics Computation:
Implements Equations 3.70 to 3.76 from the Capstone Report:
- N_params: Parameter count (Eq. 3.70)
- FPS: Frames per second = N_frames / T_inference (Eq. 3.71)
- T_latency: Average inference latency = T_total / N (Eq. 3.72)
- M_model: Model memory requirement (Eq. 3.73)
- CR: Compression Ratio = M_original / M_compressed (Eq. 3.74)
- P: Electrical power consumption = V * I (Eq. 3.75)
- E_inf: Energy consumed per inference = P * T_latency (Eq. 3.76)
"""

import os
import time
import numpy as np


def compute_edge_metrics(
    tflite_path,
    original_model_path=None,
    input_shape=(1, 128, 128, 3),
    n_runs=100,
    voltage=5.0,     # Raspberry Pi 5V power supply
    current=1.2,     # Average operating current (1.2 Amps)
):
    import tensorflow as tf

    # 1. Load TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.random.uniform(0.0, 1.0, size=input_shape).astype(np.float32)

    # Warm-up runs
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # 2. Measure Inference Latency (Eq. 3.72) and FPS (Eq. 3.71)
    start_time = time.perf_counter()
    for _ in range(n_runs):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
    total_time = time.perf_counter() - start_time

    t_latency = total_time / n_runs           # seconds per sample (Eq. 3.72)
    fps = n_runs / total_time                 # frames per second (Eq. 3.71)

    # 3. Model Size and Compression Ratio (Eq. 3.73 - 3.74)
    m_compressed = os.path.getsize(tflite_path) / (1024 * 1024)  # MB
    if original_model_path and os.path.exists(original_model_path):
        m_original = os.path.getsize(original_model_path) / (1024 * 1024)
        cr = m_original / m_compressed
    else:
        m_original = None
        cr = None

    # 4. Power & Energy Consumption (Eq. 3.75 - 3.76)
    power = voltage * current                 # Watts (Eq. 3.75)
    e_inf = power * t_latency                 # Joules per inference (Eq. 3.76)

    return {
        "tflite_model": tflite_path,
        "latency_ms": t_latency * 1000.0,
        "fps": fps,
        "compressed_size_mb": m_compressed,
        "original_size_mb": m_original,
        "compression_ratio": cr,
        "power_watts": power,
        "energy_per_inference_joules": e_inf,
    }
