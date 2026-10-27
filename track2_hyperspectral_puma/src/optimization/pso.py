"""
Particle Swarm Optimization (PSO) for spectral band selection.
"""

import time
import tracemalloc
from typing import Callable, List, Sequence
import numpy as np

from .base import BaseOptimizer, OptimizationResult, ensure_unique_bands


class ParticleSwarmOptimizer(BaseOptimizer):
    """Particle Swarm Optimization adapted for discrete band selection."""

    def __init__(
        self,
        k_bands: int,
        total_bands: int,
        pop_size: int = 30,
        max_iter: int = 50,
        random_seed: int = 42,
        w_max: float = 0.9,
        w_min: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        super().__init__(
            name="PSO",
            k_bands=k_bands,
            total_bands=total_bands,
            pop_size=pop_size,
            max_iter=max_iter,
            random_seed=random_seed,
        )
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

    def optimize(self, fitness_func: Callable[[Sequence[int]], float]) -> OptimizationResult:
        tracemalloc.start()
        start_time = time.perf_counter()

        lb = 0.0
        ub = float(self.total_bands - 1)
        v_max = (ub - lb) * 0.2

        # Initialize particles and velocities
        positions = self.rng.uniform(lb, ub, size=(self.pop_size, self.k_bands))
        velocities = self.rng.uniform(-v_max, v_max, size=(self.pop_size, self.k_bands))

        pbest_positions = positions.copy()
        pbest_bands = []
        pbest_fitness = np.zeros(self.pop_size, dtype=np.float32)

        for i in range(self.pop_size):
            bands = ensure_unique_bands(positions[i], self.total_bands, self.k_bands, self.rng)
            pbest_bands.append(bands)
            pbest_fitness[i] = self.evaluate_individual(bands, fitness_func)

        gbest_idx = np.argmin(pbest_fitness)
        gbest_pos = pbest_positions[gbest_idx].copy()
        gbest_bands = pbest_bands[gbest_idx].copy()
        gbest_fitness = float(pbest_fitness[gbest_idx])
        self.convergence = [gbest_fitness]

        for t in range(1, self.max_iter + 1):
            w = self.w_max - ((self.w_max - self.w_min) * (t / self.max_iter))

            for i in range(self.pop_size):
                r1 = self.rng.rand(self.k_bands)
                r2 = self.rng.rand(self.k_bands)

                # Velocity update
                cognitive = self.c1 * r1 * (pbest_positions[i] - positions[i])
                social = self.c2 * r2 * (gbest_pos - positions[i])
                velocities[i] = w * velocities[i] + cognitive + social
                velocities[i] = np.clip(velocities[i], -v_max, v_max)

                # Position update
                positions[i] = np.clip(positions[i] + velocities[i], lb, ub)

                # Repair to unique bands and evaluate
                candidate_bands = ensure_unique_bands(positions[i], self.total_bands, self.k_bands, self.rng)
                score = self.evaluate_individual(candidate_bands, fitness_func)

                # Update personal best
                if score < pbest_fitness[i]:
                    pbest_fitness[i] = score
                    pbest_positions[i] = positions[i].copy()
                    pbest_bands[i] = candidate_bands.copy()

                    # Update global best
                    if score < gbest_fitness:
                        gbest_fitness = float(score)
                        gbest_pos = positions[i].copy()
                        gbest_bands = candidate_bands.copy()

            self.convergence.append(gbest_fitness)

        end_time = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return OptimizationResult(
            algorithm=self.name,
            best_bands=list(gbest_bands),
            best_fitness=gbest_fitness,
            convergence=self.convergence,
            runtime_sec=end_time - start_time,
            peak_memory_mb=peak_mem / (1024**2),
            evaluations=self.eval_count,
            seed=self.random_seed,
        )
