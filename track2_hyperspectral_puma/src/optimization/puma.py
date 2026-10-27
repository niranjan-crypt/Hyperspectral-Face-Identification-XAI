"""
Puma Optimizer (PO) for spectral band selection.
Reference:
    Abdollahzadeh, B., Khodadadi, N., Barshandeh, S., Trojovský, P.,
    Gharehchopogh, F. S., El-kenawy, E. S. M., Abualigah, L., & Mirjalili, S. (2024).
    Puma optimizer (PO): a novel metaheuristic optimization algorithm and its
    application in machine learning. Cluster Computing, 27(4), 5235–5283.
    DOI: 10.1007/s10586-023-04221-5
"""

import time
import tracemalloc
from typing import Callable, Dict, List, Sequence, Tuple
import numpy as np

from .base import BaseOptimizer, OptimizationResult, ensure_unique_bands


class PumaOptimizer(BaseOptimizer):
    """
    Puma Optimizer (PO) implementing Phase Intelligence (PI),
    territorial exploration, and stalking/ambush exploitation operators.
    """

    def __init__(
        self,
        k_bands: int,
        total_bands: int,
        pop_size: int = 30,
        max_iter: int = 50,
        random_seed: int = 42,
    ):
        super().__init__(
            name="PUMA",
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

        # 1. Initialize puma population positions uniformly in [lb, ub]
        # Shape: [N, K]
        population = self.rng.uniform(lb, ub, size=(self.pop_size, self.k_bands))
        fitness = np.zeros(self.pop_size, dtype=np.float32)
        discrete_pop = []

        for i in range(self.pop_size):
            bands = ensure_unique_bands(population[i], self.total_bands, self.k_bands, self.rng)
            discrete_pop.append(bands)
            fitness[i] = self.evaluate_individual(bands, fitness_func)

        # Identify leader / best puma
        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_bands = discrete_pop[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        self.convergence = [best_fitness]

        # 2. Iterative optimization loop
        for t in range(1, self.max_iter + 1):
            # Phase Intelligence parameter: decays non-linearly to transition from exploration to exploitation
            phase_intel = 1.0 - ((t / self.max_iter) ** 2)

            # Adaptive contraction factor
            a = 2.0 * (1.0 - (t / self.max_iter))
            cf = (1.0 - (t / self.max_iter)) ** (2.0 * t / self.max_iter)

            for i in range(self.pop_size):
                p_i = self.rng.rand()
                x_new = np.empty(self.k_bands, dtype=np.float64)

                # --- Exploration Phase (Territorial Roaming & Ambush Search) ---
                if p_i < phase_intel:
                    r1 = self.rng.uniform(-1.0, 1.0, size=self.k_bands)
                    r2 = self.rng.uniform(-1.0, 1.0, size=self.k_bands)

                    # Select two distinct random pumas
                    rand_indices = self.rng.choice([idx for idx in range(self.pop_size) if idx != i], size=2, replace=False)
                    x_r1 = population[rand_indices[0]]
                    x_r2 = population[rand_indices[1]]
                    pop_mean = np.mean(population, axis=0)

                    if self.rng.rand() < 0.5:
                        # Territorial roaming across home range
                        x_new = population[i] + r1 * (x_r1 - x_r2) + r2 * (pop_mean - population[i])
                    else:
                        # Ambush search from elevated terrain (Cauchy-scaled step)
                        cauchy_step = cf * self.rng.standard_cauchy(size=self.k_bands)
                        x_new = best_pos + cauchy_step * (ub - lb) * 0.1

                # --- Exploitation Phase (Stalking & Leap Attack) ---
                else:
                    r3 = self.rng.rand(self.k_bands)
                    r4 = self.rng.rand(self.k_bands)
                    a_vec = 2.0 * a * r3 - a
                    c_vec = 2.0 * r4

                    if self.rng.rand() < 0.5:
                        # Stalking: closing in on target prey
                        dist_to_best = np.abs(c_vec * best_pos - population[i])
                        x_new = best_pos - a_vec * dist_to_best
                    else:
                        # Leap attack: spiral refinement around prey
                        r5 = self.rng.uniform(-1.0, 1.0)
                        r6 = self.rng.rand()
                        x_new = best_pos + r5 * np.cos(2.0 * np.pi * r6) * np.abs(best_pos - population[i])

                # Boundary clipping
                x_new = np.clip(x_new, lb, ub)

                # Enforce unique valid band indices
                candidate_bands = ensure_unique_bands(x_new, self.total_bands, self.k_bands, self.rng)
                new_fitness = self.evaluate_individual(candidate_bands, fitness_func)

                # Greedy selection
                if new_fitness < fitness[i]:
                    population[i] = x_new
                    discrete_pop[i] = candidate_bands
                    fitness[i] = new_fitness

                    if new_fitness < best_fitness:
                        best_fitness = float(new_fitness)
                        best_pos = x_new.copy()
                        best_bands = candidate_bands.copy()

            self.convergence.append(best_fitness)

        end_time = time.perf_counter()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return OptimizationResult(
            algorithm=self.name,
            best_bands=list(best_bands),
            best_fitness=best_fitness,
            convergence=self.convergence,
            runtime_sec=end_time - start_time,
            peak_memory_mb=peak_mem / (1024**2),
            evaluations=self.eval_count,
            seed=self.random_seed,
        )
