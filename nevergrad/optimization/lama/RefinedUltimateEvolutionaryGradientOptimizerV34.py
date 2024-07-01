import numpy as np


class RefinedUltimateEvolutionaryGradientOptimizerV34:
    def __init__(
        self,
        budget=10000,
        population_size=135,
        F_base=0.58,
        F_range=0.42,
        CR=0.93,
        elite_fraction=0.08,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Adjusted base mutation factor for better balance
        self.F_range = F_range  # Reduced range to control mutation variability
        self.CR = CR  # Adjusted crossover probability for enhanced exploration within sub-spaces
        self.elite_fraction = (
            elite_fraction  # Modified elite fraction to focus on a slightly broader elite group
        )
        self.mutation_strategy = mutation_strategy  # Adaptive strategy prioritizes exploiting promising areas
        self.dim = 5  # Dimensionality of the problem is fixed to 5 as given
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

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
                if self.mutation_strategy == "adaptive":
                    # Adaptively choose base individual from either the best or a randomly selected elite
                    if np.random.rand() < 0.8:  # Increased adaptive focus to exploit known good regions
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    # Random selection of base from elite members
                    base = population[np.random.choice(elite_indices)]

                # Adjust F dynamically within a controlled range for better convergence
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Mutation strategy DE/rand/1
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover with enhanced probability
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] is True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial individual and update if improvement
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Check for budget exhaustion
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
