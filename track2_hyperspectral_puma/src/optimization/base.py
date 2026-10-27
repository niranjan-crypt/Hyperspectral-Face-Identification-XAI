"""
Abstract Base Optimizer and common result containers.
Enforces equal objective evaluation budget, reproducible random seeds,
and strict uniqueness constraints on selected band indices.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import tracemalloc
from typing import Callable, Dict, List, Optional, Tuple, Any, Sequence
import numpy as np


@dataclass
class OptimizationResult:
    algorithm: str
    best_bands: List[int]
    best_fitness: float
    convergence: List[float]
    runtime_sec: float
    peak_memory_mb: float
    evaluations: int
    seed: int
    detailed_metrics: Dict[str, float] = field(default_factory=dict)


def ensure_unique_bands(indices: np.ndarray, total_bands: int, k: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Guarantees exactly k unique integer band indices in [0, total_bands - 1].
    Repairs any duplicates or out-of-bound indices.
    """
    clipped = np.clip(np.round(indices).astype(np.int32), 0, total_bands - 1)
    unique_set = list(dict.fromkeys(clipped))

    if len(unique_set) < k:
        all_possible = np.setdiff1d(np.arange(total_bands), unique_set)
        needed = k - len(unique_set)
        filler = rng.choice(all_possible, size=needed, replace=False)
        unique_set.extend(filler)
    elif len(unique_set) > k:
        unique_set = unique_set[:k]

    return np.array(sorted(unique_set), dtype=np.int32)


class BaseOptimizer(ABC):
    """Abstract base class for band selection optimizers."""

    def __init__(
        self,
        name: str,
        k_bands: int,
        total_bands: int,
        pop_size: int = 30,
        max_iter: int = 50,
        random_seed: int = 42,
    ):
        self.name = name
        self.k_bands = k_bands
        self.total_bands = total_bands
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        self.eval_count = 0
        self.convergence: List[float] = []

    @abstractmethod
    def optimize(self, fitness_func: Callable[[Sequence[int]], float]) -> OptimizationResult:
        """Runs the optimization algorithm under the fixed computational budget."""
        pass

    def evaluate_individual(self, bands: np.ndarray, fitness_func: Callable[[Sequence[int]], float]) -> float:
        """Helper to evaluate and increment objective count."""
        self.eval_count += 1
        return fitness_func(bands)
