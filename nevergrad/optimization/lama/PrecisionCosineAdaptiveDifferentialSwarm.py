import numpy as np


class PrecisionCosineAdaptiveDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 200  # Reduced population size to focus on quality solutions
        self.F_base = 0.85  # Increased base mutation factor for stronger mutations
        self.CR = 0.85  # Slightly reduced crossover probability to balance exploration
        self.adaptive_F_adjustment = 0.1  # Lower change rate for mutation factor
        self.top_percentile = 0.1  # Reduced to top 10% to further focus on elite solutions
        self.epsilon = 1e-10  # Small constant to avoid division by zero in cosine computation

    def __call__(self, func):
        # Initialize population within the bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main loop over the budget
        for i in range(int(self.budget / self.pop_size)):
            # Cosine adaptive mutation factor with precision control
            iteration_ratio = i / (self.budget / self.pop_size + self.epsilon)
            F_adaptive = self.F_base + self.adaptive_F_adjustment * np.cos(2 * np.pi * iteration_ratio)

            for j in range(self.pop_size):
                # Mutation strategy: DE/current-to-best/1 with cosine adaptive F
                idxs = np.argsort(fitness)[: int(self.top_percentile * self.pop_size)]  # top individuals
                best_local = pop[np.random.choice(idxs)]
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b = pop[np.random.choice(idxs, 2, replace=False)]
                mutant = pop[j] + F_adaptive * (best_local - pop[j]) + F_adaptive * (a - b)

                # Clip to ensure staying within bounds
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
