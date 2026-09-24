"""
Particle Swarm Optimization (PSO) for Hyperspectral Band Selection.
Implements Equations 3.77, 3.78, and 3.26 from the Capstone Report.
"""

import numpy as np


class PSOBandSelector:
    def __init__(
        self,
        fitness_evaluator,
        n_particles=30,
        max_iter=80,
        w=0.729,
        c1=1.494,
        c2=1.494,
        threshold=0.50,
        seed=42,
    ):
        self.evaluator = fitness_evaluator
        self.B = fitness_evaluator.B
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.threshold = threshold
        self.seed = seed

    def optimize(self):
        np.random.seed(self.seed)

        # Initialize continuous particle positions and velocities
        X = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.B))
        V = np.random.uniform(-0.1, 0.1, size=(self.n_particles, self.B))

        # Personal bests and Global best
        p_best = np.copy(X)
        p_best_scores = np.full(self.n_particles, -np.inf)

        g_best = None
        g_best_score = -np.inf

        # Evaluate initial particles
        for i in range(self.n_particles):
            selected = np.where(X[i] > self.threshold)[0]
            # Ensure within constraint (min_bands to max_bands)
            if len(selected) < self.evaluator.min_bands:
                top_idx = np.argsort(X[i])[-self.evaluator.min_bands:]
                selected = top_idx

            score = self.evaluator.evaluate_subset(selected)
            p_best_scores[i] = score
            if score > g_best_score:
                g_best_score = score
                g_best = np.copy(X[i])

        history = [g_best_score]

        for it in range(self.max_iter):
            r1 = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.B))
            r2 = np.random.uniform(0.0, 1.0, size=(self.n_particles, self.B))

            # Velocity update: v_t+1 = w*v_t + c1*r1*(p_i - x_i) + c2*r2*(g - x_i) (Eq. 3.77)
            V = (
                self.w * V
                + self.c1 * r1 * (p_best - X)
                + self.c2 * r2 * (g_best - X)
            )
            V = np.clip(V, -0.2, 0.2)

            # Position update: x_t+1 = x_t + v_t+1 (Eq. 3.78)
            X = X + V
            X = np.clip(X, 0.0, 1.0)

            # Evaluate each particle with threshold binarization (Eq. 3.26)
            for i in range(self.n_particles):
                selected = np.where(X[i] > self.threshold)[0]
                if len(selected) < self.evaluator.min_bands:
                    selected = np.argsort(X[i])[-self.evaluator.min_bands:]
                elif len(selected) > self.evaluator.max_bands:
                    selected = np.argsort(X[i])[-self.evaluator.max_bands:]

                score = self.evaluator.evaluate_subset(selected)
                if score > p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best[i] = np.copy(X[i])

                if score > g_best_score:
                    g_best_score = score
                    g_best = np.copy(X[i])

            history.append(g_best_score)

        # Extract optimal binary subset
        best_bands = np.where(g_best > self.threshold)[0]
        if len(best_bands) < self.evaluator.min_bands:
            best_bands = np.argsort(g_best)[-self.evaluator.min_bands:]
        elif len(best_bands) > self.evaluator.max_bands:
            best_bands = np.argsort(g_best)[-self.evaluator.max_bands:]

        return sorted(list(best_bands)), g_best_score, history
