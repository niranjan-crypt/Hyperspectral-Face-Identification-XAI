"""
SHAP (SHapley Additive exPlanations) for Spectral and Spatial Feature Importance.
Implements Equations 3.58 and 3.59 from the Capstone Report:
phi_i = sum_(S subseteq N\\{i}) [|S|! (|N| - |S| - 1)! / |N|!] * [f(S U {i}) - f(S)]   [Eq. 3.59]
"""

import numpy as np
try:
    import shap
except ImportError:
    shap = None


class SpectralSHAPExplainer:
    def __init__(self, predict_fn, background_data):
        """
        predict_fn: Callable that accepts (N, D) or (N, H, W, C) and returns class probabilities.
        background_data: Representative background dataset for Shapley baseline.
        """
        self.predict_fn = predict_fn
        self.background_data = background_data
        self.explainer = None
        if shap is not None:
            self.explainer = shap.KernelExplainer(predict_fn, background_data)

    def explain_instance(self, instance, nsamples=100):
        """
        Calculates Shapley values phi_i for a given input instance.
        """
        if self.explainer is None:
            raise RuntimeError("SHAP package not installed. Run: pip install shap")
        shap_values = self.explainer.shap_values(instance, nsamples=nsamples)
        return shap_values

    def compute_band_importance(self, spectral_instances, nsamples=50):
        """
        Aggregates absolute Shapley values across instances to rank spectral bands.
        """
        shap_vals = self.explain_instance(spectral_instances, nsamples=nsamples)
        if isinstance(shap_vals, list):
            # Multiclass: aggregate across classes
            importance = np.mean([np.mean(np.abs(sv), axis=0) for sv in shap_vals], axis=0)
        else:
            importance = np.mean(np.abs(shap_vals), axis=0)
        return importance
