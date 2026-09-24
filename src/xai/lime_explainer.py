"""
LIME (Local Interpretable Model-agnostic Explanations) for Face Recognition Models.
Implements Equation 3.62 from the Capstone Report:
xi(x) = argmin_(g in G) [ L(f, g, pi_x) + Omega(g) ]   [Eq. 3.62]
"""

import numpy as np
try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
except ImportError:
    lime_image = None
    mark_boundaries = None


class LIMEFaceExplainer:
    def __init__(self, mode="image"):
        self.mode = mode
        if lime_image is not None:
            self.explainer = lime_image.LimeImageExplainer()
        else:
            self.explainer = None

    def explain_image(
        self,
        img_rgb,
        predict_fn,
        top_labels=3,
        hide_color=0,
        num_samples=200,
    ):
        """
        Explains predictions of predict_fn for an RGB face image.
        Returns:
            temp: numpy array with boundaries marked
            mask: boolean mask of influential superpixels
        """
        if self.explainer is None:
            raise RuntimeError("LIME package not installed. Run: pip install lime")

        explanation = self.explainer.explain_instance(
            img_rgb.astype(np.double),
            predict_fn,
            top_labels=top_labels,
            hide_color=hide_color,
            num_samples=num_samples,
        )

        top_pred = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(
            top_pred, positive_only=True, num_features=5, hide_rest=False
        )

        marked_img = mark_boundaries(temp / 255.0, mask)
        return (marked_img * 255).astype(np.uint8), mask, top_pred
