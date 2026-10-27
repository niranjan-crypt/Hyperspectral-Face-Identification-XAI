"""
Manifold Projection and Cluster Separability Evaluation.
Computes 2D t-SNE and PCA embeddings and quantitative clustering metrics:
    - Silhouette Score (higher is better)
    - Davies-Bouldin Index (lower is better)
    - Calinski-Harabasz Index (higher is better)
"""

from typing import Dict, List, Sequence, Tuple, Any, Optional
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


class ManifoldClusterEvaluator:
    """Evaluates cluster separability of facial representations in 2D manifold space."""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed

    def evaluate_embedding(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        bands: Optional[Sequence[int]] = None,
        n_samples: int = 1500,
    ) -> Dict[str, Any]:
        """Subsamples test data and computes PCA, t-SNE, and clustering metrics."""
        X_sub = X_test if bands is None else X_test[:, list(bands)]

        rng = np.random.RandomState(self.seed)
        sample_idx = rng.choice(len(X_sub), size=min(len(X_sub), n_samples), replace=False)
        X_s = X_sub[sample_idx]
        y_s = y_test[sample_idx]

        # 1. PCA 2D
        pca = PCA(n_components=2, random_state=self.seed)
        pca_coords = pca.fit_transform(X_s)
        pca_var = float(np.sum(pca.explained_variance_ratio_))

        # 2. t-SNE 2D
        tsne = TSNE(n_components=2, perplexity=30, random_state=self.seed, max_iter=800)
        tsne_coords = tsne.fit_transform(X_s)

        # 3. Quantitative Cluster Validation Metrics
        sil = float(silhouette_score(X_s, y_s, metric="cosine"))
        db = float(davies_bouldin_score(X_s, y_s))
        ch = float(calinski_harabasz_score(X_s, y_s))

        return {
            "pca_coords": pca_coords,
            "tsne_coords": tsne_coords,
            "y": y_s,
            "pca_explained_variance": pca_var,
            "silhouette_score": sil,
            "davies_bouldin_index": db,
            "calinski_harabasz_score": ch,
        }
