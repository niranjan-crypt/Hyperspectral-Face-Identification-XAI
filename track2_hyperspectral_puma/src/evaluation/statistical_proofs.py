"""
Statistical Hypothesis Testing and Mathematical Proof Engine.
Includes:
    - Wilcoxon Signed-Rank Test (paired non-parametric test)
    - Friedman Multi-Algorithm Ranking Test
    - Holm-Bonferroni correction for family-wise error rate
    - Multi-dimensional Fisher Criterion J = Tr(Sw^-1 * Sb)
    - Matrix Condition Number kappa(R)
    - Average Pairwise Mutual Information (MI)
"""

from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.feature_selection import mutual_info_classif


class StatisticalProofEngine:
    """Computes peer-reviewed statistical hypothesis tests and information-theoretic proofs."""

    @staticmethod
    def compute_wilcoxon_tests(
        puma_values: List[float],
        competitors: Dict[str, List[float]],
        alternative: str = "less",  # 'less' for fitness (minimization), 'greater' for accuracy
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculates pairwise Wilcoxon signed-rank tests comparing PUMA against each benchmark.
        Returns test statistic, raw p-value, and Holm-Bonferroni adjusted p-value.
        """
        results = {}
        p_values = []
        algo_names = list(competitors.keys())

        puma_arr = np.array(puma_values)
        for name in algo_names:
            comp_arr = np.array(competitors[name])
            diff = puma_arr - comp_arr

            # Check if all differences are zero
            if np.all(diff == 0):
                stat, p_val = 0.0, 1.0
            else:
                try:
                    res = wilcoxon(puma_arr, comp_arr, alternative=alternative)
                    stat, p_val = float(res.statistic), float(res.pvalue)
                except Exception:
                    stat, p_val = 0.0, 1.0

            results[name] = {"statistic": stat, "raw_p": p_val}
            p_values.append(p_val)

        # Holm-Bonferroni step-down adjustment
        sorted_indices = np.argsort(p_values)
        m = len(p_values)
        adjusted_p = np.zeros(m)

        for rank, idx in enumerate(sorted_indices):
            multiplier = m - rank
            p_adj = min(1.0, p_values[idx] * multiplier)
            adjusted_p[idx] = p_adj

        # Enforce monotonicity
        for i in range(1, m):
            adjusted_p[sorted_indices[i]] = max(adjusted_p[sorted_indices[i]], adjusted_p[sorted_indices[i - 1]])

        for i, name in enumerate(algo_names):
            results[name]["adjusted_p"] = float(adjusted_p[i])
            results[name]["significant_05"] = bool(adjusted_p[i] < 0.05)
            results[name]["significant_01"] = bool(adjusted_p[i] < 0.01)

        return results

    @staticmethod
    def compute_friedman_test(metrics_by_algo: Dict[str, List[float]], minimize: bool = True) -> Dict[str, Any]:
        """
        Computes the non-parametric Friedman test for multiple algorithm comparisons.
        Calculates average ranks (Rank 1 = best).
        """
        algos = list(metrics_by_algo.keys())
        matrix = np.column_stack([metrics_by_algo[a] for a in algos])  # [N_runs, N_algos]

        # Invert if minimizing so that lower values get lower rank (rank 1)
        ranks = np.zeros_like(matrix)
        for row_idx in range(len(matrix)):
            row = matrix[row_idx]
            order = np.argsort(row) if minimize else np.argsort(-row)
            row_ranks = np.empty_like(order, dtype=float)
            row_ranks[order] = np.arange(1, len(row) + 1)
            ranks[row_idx] = row_ranks

        mean_ranks = np.mean(ranks, axis=0)

        # Friedman test
        stat, p_val = friedmanchisquare(*[matrix[:, i] for i in range(len(algos))])

        rank_dict = {algos[i]: float(mean_ranks[i]) for i in range(len(algos))}
        # Sort by rank ascending
        sorted_ranks = dict(sorted(rank_dict.items(), key=lambda item: item[1]))

        return {
            "friedman_stat": float(stat),
            "p_value": float(p_val),
            "average_ranks": sorted_ranks,
            "best_ranked_algorithm": list(sorted_ranks.keys())[0],
        }

    @staticmethod
    def compute_fisher_multivariate_criterion(X: np.ndarray, y: np.ndarray, bands: Sequence[int]) -> float:
        """
        Computes the multi-dimensional Fisher Scatter Criterion:
            J = Tr(Sw^-1 * Sb)
        Measures the ratio of between-class separation to within-class dispersion in K-dimensional space.
        Higher J indicates mathematically superior class separability.
        """
        X_sub = X[:, list(bands)]
        classes = np.unique(y)
        k = len(bands)

        global_mean = np.mean(X_sub, axis=0, keepdims=True)
        Sb = np.zeros((k, k), dtype=np.float64)
        Sw = np.zeros((k, k), dtype=np.float64)

        for c in classes:
            X_c = X_sub[y == c]
            n_c = len(X_c)
            c_mean = np.mean(X_c, axis=0, keepdims=True)
            diff_mean = c_mean - global_mean
            Sb += n_c * np.dot(diff_mean.T, diff_mean)

            diff_samples = X_c - c_mean
            Sw += np.dot(diff_samples.T, diff_samples)

        # Regularize Sw for numerical stability
        Sw += np.eye(k) * 1e-4 * np.trace(Sw)

        try:
            Sw_inv = np.linalg.inv(Sw)
            J = float(np.trace(np.dot(Sw_inv, Sb)))
        except np.linalg.LinAlgError:
            J = 0.0

        return J

    @staticmethod
    def compute_information_theoretic_metrics(X: np.ndarray, y: np.ndarray, bands: Sequence[int]) -> Dict[str, float]:
        """
        Calculates information-theoretic metrics:
            - Mean Mutual Information with Identity Labels: I(b_i; Y)
            - Inter-band Mutual Information: I(b_i; b_j)
            - Correlation Matrix Condition Number: kappa(R)
        """
        X_sub = X[:, list(bands)]
        k = len(bands)

        # 1. Mutual information of each band with subject identity
        # Subsample for computational speed
        n_samples = min(len(X_sub), 4000)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_sub), size=n_samples, replace=False)

        mi_identity = mutual_info_classif(X_sub[idx], y[idx], random_state=42)
        mean_mi_identity = float(np.mean(mi_identity))

        # 2. Condition number of correlation matrix
        corr = np.corrcoef(X_sub[idx].T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals, 1e-7)
        condition_number = float(np.max(eigvals) / np.min(eigvals))

        # 3. Off-diagonal correlation
        if k > 1:
            triu = np.triu_indices(k, k=1)
            mean_corr = float(np.mean(np.abs(corr[triu])))
        else:
            mean_corr = 0.0

        return {
            "mean_mi_with_identity": mean_mi_identity,
            "condition_number": condition_number,
            "mean_interband_correlation": mean_corr,
        }
