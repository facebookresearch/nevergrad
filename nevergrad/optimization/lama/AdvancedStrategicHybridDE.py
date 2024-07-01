import numpy as np


class AdvancedStrategicHybridDE:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        F_base=0.6,
        F_range=0.4,
        CR=0.95,
        hybridization_factor=0.25,
        elite_strategy=True,
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.F_range = F_range  # Range to vary F for diversity
        self.CR = CR  # Crossover probability
        self.hybridization_factor = hybridization_factor  # Factor for hybrid mutation strategy
        self.elite_strategy = elite_strategy  # Use elite strategy to focus on top performers
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
                if self.elite_strategy and np.random.rand() < 0.1:
                    # Use one of the top 10% performers as the base
                    elite_candidates = np.argsort(fitness)[: max(1, self.population_size // 10)]
                    base = population[np.random.choice(elite_candidates)]
                elif np.random.rand() < self.hybridization_factor:
                    # Occasionally use the best individual for mutation base
                    base = best_individual
                else:
                    # Regular selection excluding self
                    base = population[
                        np.random.choice([idx for idx in range(self.population_size) if idx != i])
                    ]

                # Dynamically adjust F
                F = self.F_base + np.random.rand() * self.F_range

                # Mutation using differential evolution strategy
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
