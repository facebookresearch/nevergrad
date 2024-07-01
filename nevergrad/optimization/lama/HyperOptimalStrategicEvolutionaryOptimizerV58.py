import numpy as np


class HyperOptimalStrategicEvolutionaryOptimizerV58:
    def __init__(
        self,
        budget=10000,
        population_size=138,
        F_base=0.53,
        F_range=0.47,
        CR=0.93,
        elite_fraction=0.08,
        mutation_strategy="dynamic",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Fine-tuned base mutation factor
        self.F_range = F_range  # Expanded mutation range for better exploration
        self.CR = CR  # Crossover probability adjusted for improved convergence
        self.elite_fraction = elite_fraction  # Reduced elite fraction for maintaining diversity
        self.mutation_strategy = mutation_strategy  # Dynamic mutation strategy for responsive adaptation
        self.dim = 5  # Dimensionality is set to 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population uniformly within the search bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Main optimization loop
        while evaluations < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                # Select mutation base dynamically
                if np.random.rand() < 0.85:  # Increased probability to select the best individual
                    base = best_individual
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamically adjust F within a broader range
                F = (
                    self.F_base + np.sin(np.pi * np.random.rand()) * self.F_range
                )  # Using sine modulation for F

                # Mutation using DE's rand/1 strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial solution
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Exit if budget is reached
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
