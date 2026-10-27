"""
Genetic Algorithm (GA) for Hyperspectral Band Selection.
Implements Equations 3.25 to 3.31 from the Capstone Report:
- Identity-guided continuous population initialization (Eq. 3.25)
- Opposition-based learning (Eq. 3.28 - 3.29)
- Identity-score-dependent adaptive mutation probability (Eq. 3.30 - 3.31)
"""

import numpy as np


class GeneticBandSelector:
    def __init__(
        self,
        fitness_evaluator,
        pop_size=40,
        generations=60,
        crossover_rate=0.80,
        p_min=0.02,
        p_max=0.20,
        alpha=0.60,
        threshold=0.50,
        seed=42,
    ):
        self.evaluator = fitness_evaluator
        self.B = fitness_evaluator.B
        self.pop_size = pop_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.p_min = p_min
        self.p_max = p_max
        self.alpha = alpha
        self.threshold = threshold
        self.seed = seed

    def _initialize_population(self):
        """
        Initializes population using identity-guidance:
        x_ib = alpha * I_b + (1 - alpha) * U(0, 1)   [Equation 3.25]
        """
        I_b = self.evaluator.identity_scores  # shape (B,)
        pop = np.zeros((self.pop_size, self.B), dtype=np.float32)

        for i in range(self.pop_size):
            u = np.random.uniform(0.0, 1.0, size=self.B)
            x_i = self.alpha * I_b + (1.0 - self.alpha) * u

            # Opposition-based learning (Eq. 3.28 - 3.29)
            x_opp = 1.0 - x_i
            s_orig = np.where(x_i > self.threshold)[0]
            s_opp = np.where(x_opp > self.threshold)[0]

            f_orig = self.evaluator.evaluate_subset(s_orig)
            f_opp = self.evaluator.evaluate_subset(s_opp)

            pop[i] = x_i if f_orig >= f_opp else x_opp

        return pop

    def optimize(self):
        np.random.seed(self.seed)
        pop = self._initialize_population()

        fitness_scores = np.zeros(self.pop_size)
        g_best = None
        g_best_score = -np.inf
        history = []

        I_b = self.evaluator.identity_scores
        # Adaptive mutation probability per band: p_b = p_min + (p_max - p_min)*(1 - I_b) (Eq. 3.30)
        p_mut = self.p_min + (self.p_max - self.p_min) * (1.0 - I_b)

        for gen in range(self.generations):
            for i in range(self.pop_size):
                selected = np.where(pop[i] > self.threshold)[0]
                if len(selected) < self.evaluator.min_bands:
                    selected = np.argsort(pop[i])[-self.evaluator.min_bands:]
                elif len(selected) > self.evaluator.max_bands:
                    selected = np.argsort(pop[i])[-self.evaluator.max_bands:]

                score = self.evaluator.evaluate_subset(selected)
                fitness_scores[i] = score

                if score > g_best_score:
                    g_best_score = score
                    g_best = np.copy(pop[i])

            history.append(g_best_score)

            # Tournament selection
            new_pop = []
            for _ in range(self.pop_size // 2):
                i1, i2 = np.random.choice(self.pop_size, 2, replace=False)
                parent1 = pop[i1] if fitness_scores[i1] >= fitness_scores[i2] else pop[i2]

                i3, i4 = np.random.choice(self.pop_size, 2, replace=False)
                parent2 = pop[i3] if fitness_scores[i3] >= fitness_scores[i4] else pop[i4]

                # Uniform Crossover
                if np.random.rand() < self.crossover_rate:
                    mask = np.random.rand(self.B) > 0.5
                    child1 = np.where(mask, parent1, parent2)
                    child2 = np.where(mask, parent2, parent1)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Adaptive Gaussian Mutation (Eq. 3.31)
                mut_mask1 = np.random.rand(self.B) < p_mut
                child1 += mut_mask1 * np.random.normal(0, 0.1, size=self.B)

                mut_mask2 = np.random.rand(self.B) < p_mut
                child2 += mut_mask2 * np.random.normal(0, 0.1, size=self.B)

                new_pop.extend([np.clip(child1, 0, 1), np.clip(child2, 0, 1)])

            pop = np.array(new_pop)

        best_bands = np.where(g_best > self.threshold)[0]
        if len(best_bands) < self.evaluator.min_bands:
            best_bands = np.argsort(g_best)[-self.evaluator.min_bands:]
        elif len(best_bands) > self.evaluator.max_bands:
            best_bands = np.argsort(g_best)[-self.evaluator.max_bands:]

        return sorted(list(best_bands)), g_best_score, history
