"""
Differential Evolution (DE) for spectral band selection.
"""

import time
import tracemalloc
from typing import Callable, List, Sequence
import numpy as np

from .base import BaseOptimizer, OptimizationResult, ensure_unique_bands


class DifferentialEvolution(BaseOptimizer):
    """Differential Evolution (DE/rand/1/bin) adapted for band selection."""

    def __init__(
        self,
        k_bands: int,
        total_bands: int,
        pop_size: int = 30,
        max_iter: int = 50,
        random_seed: int = 42,
        mutation_factor: float = 0.5,
        crossover_rate: float = 0.7,
    ):
        super().__init__(
            name="DE",
            k_bands=k_bands,
            total_bands=total_bands,
            pop_size=pop_size,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        self.F = mutation_factor
        self.CR = crossover_rate

    def optimize(self, fitness_func: Callable[[Sequence[int]], float]) -> OptimizationResult:
        tracemalloc.start()
        start_time = time.perf_counter()

        lb = 0.0
        ub = float(self.total_bands - 1)

        # Initialize continuous vectors
        population = self.rng.uniform(lb, ub, size=(self.pop_size, self.k_bands))
        fitness = np.zeros(self.pop_size, dtype=np.float32)
        discrete_pop = []

        for i in range(self.pop_size):
            bands = ensure_unique_bands(population[i], self.total_bands, self.k_bands, self.rng)
            discrete_pop.append(bands)
            fitness[i] = self.evaluate_individual(bands, fitness_func)

        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_bands = discrete_pop[best_idx].copy()
        best_fitness = float(fitness[best_idx])
        self.convergence = [best_fitness]

        for gen in range(1, self.max_iter + 1):
            for i in range(self.pop_size):
                # Pick three random individuals different from i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.rng.choice(candidates, size=3, replace=False)

                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                mutant = population[r1] + self.F * (population[r2] - population[r3])
                mutant = np.clip(mutant, lb, ub)

                # Binomial Crossover
                cross_mask = self.rng.rand(self.k_bands) < self.CR
                # Ensure at least one parameter mutated
                cross_mask[self.rng.randint(0, self.k_bands)] = True
                trial = np.where(cross_mask, mutant, population[i])

                trial_bands = ensure_unique_bands(trial, self.total_bands, self.k_bands, self.rng)
                trial_fitness = self.evaluate_individual(trial_bands, fitness_func)

                # Selection (one-to-one replacement)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    discrete_pop[i] = trial_bands
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = float(trial_fitness)
                        best_pos = trial.copy()
                        best_bands = trial_bands.copy()

            self.convergence.append(best_fitness)

        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
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
