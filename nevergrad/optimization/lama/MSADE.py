import numpy as np


class MSADE:
    def __init__(
        self, budget, population_size=100, F_base=0.6, CR_base=0.7, alpha=0.1, strategy_proportion=0.2
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base mutation factor
        self.CR_base = CR_base  # Base crossover probability
        self.alpha = alpha  # Rate of adaptive adjustment
        self.strategy_proportion = strategy_proportion  # Proportion of population to apply secondary strategy

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)
            F = self.F_base + self.alpha * np.random.randn()  # Add small noise for diversification
            CR = self.CR_base + self.alpha * np.random.randn()

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Select mutation strategy based on strategy proportion
                if np.random.rand() < self.strategy_proportion:
                    # DE/current-to-best/1
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b = population[np.random.choice(idxs, 2, replace=False)]
                    mutant = population[i] + F * (best_individual - population[i]) + F * (a - b)
                else:
                    # DE/rand/1
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + F * (b - c)

                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            population = new_population.copy()

        return best_fitness, best_individual
