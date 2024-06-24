import numpy as np


class RefinedEnhancedPrecisionEvolutionaryOptimizerV40:
    def __init__(
        self,
        budget=10000,
        population_size=135,
        F_base=0.6,
        F_range=0.35,
        CR=0.97,
        elite_fraction=0.15,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Adjusted base mutation factor for enhanced exploration
        self.F_range = F_range  # Refined range for mutation factor to prioritize stable convergence
        self.CR = CR  # Increased crossover probability to enhance offspring quality
        self.elite_fraction = elite_fraction  # Increased elite fraction for more robust elitism
        self.mutation_strategy = mutation_strategy  # Adaptive mutation strategy with refined parameters
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population within bounds
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
                # Enhanced adaptive mutation strategy with adjusted base selection probability
                if self.mutation_strategy == "adaptive":
                    if np.random.rand() < 0.8:  # Higher usage of the leading individual
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]

                F = self.F_base + (np.random.rand() - 0.5) * 2 * self.F_range  # Dynamically adjusted F

                # DE/rand/1 mutation
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Refined binomial crossover technique
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Fitness evaluation and selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
