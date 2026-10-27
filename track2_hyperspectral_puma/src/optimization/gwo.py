"""
Grey Wolf Optimizer (GWO) for spectral band selection.
"""

import time
import tracemalloc
from typing import Callable, List, Sequence
import numpy as np

from .base import BaseOptimizer, OptimizationResult, ensure_unique_bands


class GreyWolfOptimizer(BaseOptimizer):
    """Grey Wolf Optimizer guided by alpha, beta, and delta wolves."""

    def __init__(
        self,
        k_bands: int,
        total_bands: int,
        pop_size: int = 30,
        max_iter: int = 50,
        random_seed: int = 42,
    ):
        super().__init__(
            name="GWO",
            k_bands=k_bands,
            total_bands=total_bands,
            pop_size=pop_size,
            max_iter=max_iter,
            random_seed=random_seed,
        )

    def optimize(self, fitness_func: Callable[[Sequence[int]], float]) -> OptimizationResult:
        tracemalloc.start()
        start_time = time.perf_counter()

        lb = 0.0
        ub = float(self.total_bands - 1)

        # Initialize wolves positions
        positions = self.rng.uniform(lb, ub, size=(self.pop_size, self.k_bands))
        fitness = np.zeros(self.pop_size, dtype=np.float32)
        discrete_pop = []

        for i in range(self.pop_size):
            bands = ensure_unique_bands(positions[i], self.total_bands, self.k_bands, self.rng)
            discrete_pop.append(bands)
            fitness[i] = self.evaluate_individual(bands, fitness_func)

        # Identify Alpha, Beta, Delta wolves
        sorted_indices = np.argsort(fitness)
        alpha_pos = positions[sorted_indices[0]].copy()
        alpha_bands = discrete_pop[sorted_indices[0]].copy()
        alpha_score = float(fitness[sorted_indices[0]])

        beta_pos = positions[sorted_indices[1]].copy()
        beta_score = float(fitness[sorted_indices[1]])

        delta_pos = positions[sorted_indices[2]].copy()
        delta_score = float(fitness[sorted_indices[2]])

        self.convergence = [alpha_score]

        for t in range(1, self.max_iter + 1):
            # Linearly decreasing parameter a from 2 to 0
            a = 2.0 * (1.0 - (t / self.max_iter))

            for i in range(self.pop_size):
                # Update based on Alpha
                r1 = self.rng.rand(self.k_bands)
                r2 = self.rng.rand(self.k_bands)
                A1 = 2.0 * a * r1 - a
                C1 = 2.0 * r2
                D_alpha = np.abs(C1 * alpha_pos - positions[i])
                X1 = alpha_pos - A1 * D_alpha

                # Update based on Beta
                r1 = self.rng.rand(self.k_bands)
                r2 = self.rng.rand(self.k_bands)
                A2 = 2.0 * a * r1 - a
                C2 = 2.0 * r2
                D_beta = np.abs(C2 * beta_pos - positions[i])
                X2 = beta_pos - A2 * D_beta

                # Update based on Delta
                r1 = self.rng.rand(self.k_bands)
                r2 = self.rng.rand(self.k_bands)
                A3 = 2.0 * a * r1 - a
                C3 = 2.0 * r2
                D_delta = np.abs(C3 * delta_pos - positions[i])
                X3 = delta_pos - A3 * D_delta

                # Position vector averaging
                new_pos = np.clip((X1 + X2 + X3) / 3.0, lb, ub)
                candidate_bands = ensure_unique_bands(new_pos, self.total_bands, self.k_bands, self.rng)
                score = self.evaluate_individual(candidate_bands, fitness_func)

                positions[i] = new_pos
                fitness[i] = score

                # Update Alpha, Beta, Delta
                if score < alpha_score:
                    alpha_score = float(score)
                    alpha_pos = new_pos.copy()
                    alpha_bands = candidate_bands.copy()
                elif alpha_score < score < beta_score:
                    beta_score = float(score)
                    beta_pos = new_pos.copy()
                elif beta_score < score < delta_score:
                    delta_score = float(score)
                    delta_pos = new_pos.copy()

            self.convergence.append(alpha_score)

        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return OptimizationResult(
            algorithm=self.name,
            best_bands=list(alpha_bands),
            best_fitness=alpha_score,
            convergence=self.convergence,
            runtime_sec=end_time - start_time,
            peak_memory_mb=peak_mem / (1024**2),
            evaluations=self.eval_count,
            seed=self.random_seed,
        )
