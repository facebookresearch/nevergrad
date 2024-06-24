import numpy as np


class UltraEnhancedDynamicDE:
    def __init__(
        self, budget=10000, population_size=100, F_base=0.5, F_adapt=0.3, CR=0.95, adapt_strategy=True
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Factor for mutation scale
        self.F_adapt = F_adapt  # Adaptation factor for mutation scale
        self.CR = CR  # Crossover probability
        self.adapt_strategy = adapt_strategy  # Adaptation strategy toggle
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Main loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation strategy based on phase of optimization process
                phase_ratio = evaluations / self.budget
                if self.adapt_strategy and phase_ratio < 0.5:
                    idxs = np.argsort(fitness)[:2]  # Use best individuals early on
                    base = population[idxs[np.random.randint(2)]]
                else:
                    base = population[
                        np.random.choice([idx for idx in range(self.population_size) if idx != i])
                    ]

                # Dynamically adjust mutation factor F
                F = self.F_base + np.sin(phase_ratio * np.pi) * self.F_adapt

                # Mutation using derivative of best
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover using binomial method
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection step
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Exit if budget exhausted
                if evaluations >= self.budget:
                    break

        # Return the best solution found
        return best_fitness, best_individual
