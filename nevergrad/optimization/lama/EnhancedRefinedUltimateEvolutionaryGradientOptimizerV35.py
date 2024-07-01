import numpy as np


class EnhancedRefinedUltimateEvolutionaryGradientOptimizerV35:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.54,
        F_range=0.46,
        CR=0.96,
        elite_fraction=0.12,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Fine-tuned base mutation factor for improved exploration/exploitation balance
        self.F_range = (
            F_range  # Adjusted range for mutation factor to foster more aggressive explorations when needed
        )
        self.CR = CR  # Increased crossover probability to enhance trial vector diversity
        self.elite_fraction = (
            elite_fraction  # Increased elite fraction to leverage a broader base of good solutions
        )
        self.mutation_strategy = (
            mutation_strategy  # Maintaining 'adaptive' strategy for flexibility in mutation base selection
        )
        self.dim = 5  # Problem dimensionality
        self.lb = -5.0  # Lower boundary of search space
        self.ub = 5.0  # Upper boundary of search space

    def __call__(self, func):
        # Initialize population uniformly within bounds
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
                # Select mutation base according to strategy
                if self.mutation_strategy == "adaptive":
                    # Enhanced adaptive focus: adjust probability to 0.80 for selecting the best individual
                    if np.random.rand() < 0.80:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamically adjust F within given range
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Mutation (DE/rand/1)
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation with slightly increased CR
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate and select the better solution
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Check budget status
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
