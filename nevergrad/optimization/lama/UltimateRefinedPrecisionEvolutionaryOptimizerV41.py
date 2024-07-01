import numpy as np


class UltimateRefinedPrecisionEvolutionaryOptimizerV41:
    def __init__(
        self,
        budget=10000,
        population_size=130,
        F_base=0.58,
        F_range=0.42,
        CR=0.93,
        elite_fraction=0.12,
        mutation_strategy="hybrid",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly adjusted mutation factor for better global exploration
        self.F_range = F_range  # Adjusted mutation factor range for more controlled mutations
        self.CR = CR  # Slightly lower crossover probability to prevent premature convergence
        self.elite_fraction = elite_fraction  # Adjusted elite fraction to maintain a balance between exploration and exploitation
        self.mutation_strategy = (
            mutation_strategy  # Hybrid mutation strategy incorporating both random and best elements
        )
        self.dim = 5  # Dimensionality of the problem fixed at 5
        self.lb = -5.0  # Search space lower bound
        self.ub = 5.0  # Search space upper bound

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Optimization loop
        while evaluations < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                if self.mutation_strategy == "hybrid":
                    # Choose base individual: adaptive choice between best and random elite
                    if np.random.rand() < 0.80:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Calculate F dynamically within refined constraints
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Differential evolution mutation strategy (DE/rand/1)
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation with refined CR
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

                # Exit if budget is exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
