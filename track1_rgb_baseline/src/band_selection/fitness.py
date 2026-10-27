"""
Multi-Objective Fitness Formulation for Hyperspectral Band Selection.
Implements Equations 3.8 to 3.23 and Equation 3.79 from the Capstone Report:
- Identity discrimination score (Fisher's criterion)
- Spectral complementarity
- Spectral diversity with exponential distance decay
- Spectral redundancy penalty
- Sparsity regularization
- Subset stability across runs
"""

import numpy as np


class MultiObjectiveBandFitness:
    def __init__(
        self,
        X_cubes,
        y_labels,
        total_bands=1200,
        min_bands=5,
        max_bands=20,
        gamma=0.25,
        delta=0.20,
        lambda_redundancy=0.30,
        eta_sparsity=0.15,
        xi_stability=0.10,
        tau=0.15,
        epsilon=1e-6,
    ):
        """
        X_cubes: list/array of shape (N, H, W, B) or (N, H*W, B)
        y_labels: array of shape (N,)
        """
        self.N = len(y_labels)
        self.B = total_bands
        self.min_bands = min_bands
        self.max_bands = max_bands
        self.gamma = gamma
        self.delta = delta
        self.lambda_red = lambda_redundancy
        self.eta = eta_sparsity
        self.xi = xi_stability
        self.tau = tau
        self.epsilon = epsilon

        # Flatten spatial dimensions per cube to shape (N, pixels, B)
        if len(X_cubes.shape) == 4:
            N, H, W, B = X_cubes.shape
            self.X_flat = X_cubes.reshape(N, H * W, B)
        else:
            self.X_flat = X_cubes

        self.y = np.array(y_labels)
        self.classes = np.unique(self.y)
        self.C = len(self.classes)

        # Precompute per-band identity discrimination score I_b (Eq. 3.10)
        self.identity_scores = self._precompute_identity_scores()

        # Precompute band-to-band correlation matrix rho_ij (Eq. 3.11)
        self.corr_matrix = self._precompute_correlation_matrix()

    def _precompute_identity_scores(self):
        """
        Computes between-class scatter S_B and within-class scatter S_W
        for each band, then calculates Fisher's criterion score I_b (Eqs. 3.8 - 3.10).
        """
        I = np.zeros(self.B, dtype=np.float32)
        # Mean spectral vector per sample: shape (N, B)
        sample_means = np.mean(self.X_flat, axis=1)
        global_mean = np.mean(sample_means, axis=0)  # shape (B,)

        for b in range(self.B):
            sb_b = 0.0
            sw_b = 0.0
            for c in self.classes:
                c_samples = sample_means[self.y == c, b]
                n_c = len(c_samples)
                if n_c == 0:
                    continue
                mu_cb = np.mean(c_samples)

                # Between-class scatter (Eq. 3.8)
                sb_b += n_c * ((mu_cb - global_mean[b]) ** 2)

                # Within-class scatter (Eq. 3.9)
                sw_b += np.sum((c_samples - mu_cb) ** 2)

            # Discrimination score I_b (Eq. 3.10)
            I[b] = sb_b / (sw_b + self.epsilon)

        # Normalize scores to [0, 1]
        denom = (np.max(I) - np.min(I)) + self.epsilon
        return (I - np.min(I)) / denom

    def _precompute_correlation_matrix(self):
        """
        Precomputes Pearson correlation between bands across spatial-sample values (Eq. 3.11).
        """
        # Average spectra across samples: shape (N, B)
        sample_means = np.mean(self.X_flat, axis=1)
        corr = np.corrcoef(sample_means, rowvar=False)
        return np.nan_to_num(corr, nan=0.0)

    def evaluate_subset(self, subset_indices, stability_score=1.0):
        """
        Calculates composite fitness F(S) for selected subset S (Eq. 3.23 / 3.79).
        """
        S = list(set(subset_indices))
        k = len(S)

        # Constraint check (Eq. 3.19)
        if k < self.min_bands:
            return -1e5 * (self.min_bands - k)
        if k > self.max_bands:
            return -1e5 * (k - self.max_bands)

        # 1. Average Identity Discrimination D_ID(S) (Eq. 3.22)
        d_id = np.mean([self.identity_scores[b] for b in S])

        # 2. Spectral Complementarity P(S) (Eq. 3.13)
        comp_sum = 0.0
        for i_idx, i in enumerate(S):
            for j_idx in range(i_idx + 1, k):
                j = S[j_idx]
                comp_sum += (1.0 - abs(self.corr_matrix[i, j]))
        p_s = np.sum([self.identity_scores[b] for b in S]) + comp_sum / max(1, (k * (k - 1) / 2))

        # 3. Spectral Diversity D_spec(S) (Eqs. 3.14 - 3.16)
        div_sum = 0.0
        for i_idx, i in enumerate(S):
            for j_idx in range(i_idx + 1, k):
                j = S[j_idx]
                d_ij = abs(i - j) / float(self.B)
                div_sum += (1.0 - np.exp(-d_ij / self.tau))
        d_spec = div_sum / max(1, (k * (k - 1) / 2))

        # 4. Average Spectral Redundancy R(S) (Eq. 3.17)
        red_sum = 0.0
        for i_idx, i in enumerate(S):
            for j_idx in range(i_idx + 1, k):
                j = S[j_idx]
                red_sum += abs(self.corr_matrix[i, j])
        r_s = (2.0 / (k * (k - 1))) * red_sum if k > 1 else 0.0

        # 5. Sparsity term L_sparse(S) (Eq. 3.18)
        l_sparse = float(k) / float(self.B)

        # 6. Overall Multi-Objective Fitness (Eq. 3.23)
        fitness = (
            d_id
            + (self.gamma * p_s)
            + (self.delta * d_spec)
            - (self.lambda_red * r_s)
            - (self.eta * l_sparse)
            + (self.xi * stability_score)
        )
        return float(fitness)
