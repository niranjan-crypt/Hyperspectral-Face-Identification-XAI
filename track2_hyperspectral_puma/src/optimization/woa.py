"""
Whale Optimization Algorithm (WOA) for spectral band selection.
Reference:
    Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
    Advances in Engineering Software, 95, 51-71.
    DOI: 10.1016/j.advengsoft.2016.01.008
"""

import time
import tracemalloc
from typing import Callable, List, Sequence
import numpy as np

from .base import BaseOptimizer, OptimizationResult, ensure_unique_bands


class WhaleOptimizationAlgorithm(BaseOptimizer):
    """
    Whale Optimization Algorithm (WOA) modeling encircling prey,
    spiral bubble-net attacking, and exploration search mechanisms.
    """

    def __init__(
        self,
        k_bands: int,
        total_bands: int,
        pop_size: int = 30,
        max_iter: int = 50,
        random_seed: int = 42,
        b: float = 1.0,
    ):
        super().__init__(
            name="WOA",
            k_bands=k_bands,
            total_bands=total_bands,
            pop_size=pop_size,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        self.b = b

    def optimize(self, fitness_func: Callable[[Sequence[int]], float]) -> OptimizationResult:
        tracemalloc.start()
        start_time = time.perf_counter()

        lb = 0.0
        ub = float(self.total_bands - 1)

        # 1. Initialize whale population positions in [lb, ub]
        positions = self.rng.uniform(lb, ub, size=(self.pop_size, self.k_bands))
        fitness = np.zeros(self.pop_size, dtype=np.float32)
        discrete_pop = []

        for i in range(self.pop_size):
            bands = ensure_unique_bands(positions[i], self.total_bands, self.k_bands, self.rng)
            discrete_pop.append(bands)
            fitness[i] = self.evaluate_individual(bands, fitness_func)

        # Identify leader / best whale
        best_idx = np.argmin(fitness)
        leader_pos = positions[best_idx].copy()
        leader_bands = discrete_pop[best_idx].copy()
        leader_score = float(fitness[best_idx])
        self.convergence = [leader_score]

        # 2. Iteration loop
        for t in range(1, self.max_iter + 1):
            # Linearly decreasing parameter a from 2 down to 0
            a = 2.0 * (1.0 - (t / self.max_iter))

            for i in range(self.pop_size):
                r1 = self.rng.rand(self.k_bands)
                r2 = self.rng.rand(self.k_bands)

                A = 2.0 * a * r1 - a
                C = 2.0 * r2

                p = self.rng.rand()
                l = self.rng.uniform(-1.0, 1.0)

                x_new = np.empty(self.k_bands, dtype=np.float64)

                if p < 0.5:
                    if np.linalg.norm(A) < 1.0:
                        # Encircling prey (exploitation)
                        D = np.abs(C * leader_pos - positions[i])
                        x_new = leader_pos - A * D
                    else:
                        # Search for prey (exploration)
                        rand_idx = self.rng.choice([idx for idx in range(self.pop_size) if idx != i])
                        x_rand = positions[rand_idx]
                        D = np.abs(C * x_rand - positions[i])
                        x_new = x_rand - A * D
                else:
                    # Spiral bubble-net attack (exploitation)
                    distance_to_leader = np.abs(leader_pos - positions[i])
                    spiral = distance_to_leader * np.exp(self.b * l) * np.cos(2.0 * np.pi * l)
                    x_new = spiral + leader_pos

                # Boundary clipping
                x_new = np.clip(x_new, lb, ub)

                # Repair to unique valid band indices
                candidate_bands = ensure_unique_bands(x_new, self.total_bands, self.k_bands, self.rng)
                new_fitness = self.evaluate_individual(candidate_bands, fitness_func)

                # Greedy selection
                if new_fitness < fitness[i]:
                    positions[i] = x_new
                    discrete_pop[i] = candidate_bands
                    fitness[i] = new_fitness

                    if new_fitness < leader_score:
                        leader_score = float(new_fitness)
                        leader_pos = x_new.copy()
                        leader_bands = candidate_bands.copy()

            self.convergence.append(leader_score)

        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return OptimizationResult(
            algorithm=self.name,
            best_bands=list(leader_bands),
            best_fitness=leader_score,
            convergence=self.convergence,
            runtime_sec=end_time - start_time,
            peak_memory_mb=peak_mem / (1024**2),
            evaluations=self.eval_count,
            seed=self.random_seed,
        )
