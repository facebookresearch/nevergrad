import numpy as np


class ERADE:
    def __init__(
        self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50, F=0.8, CR=0.9
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability

    def __call__(self, func):
        # Initialization of population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        evaluations = self.population_size

        # Enhanced strategy adjustments
        F_min, F_max = 0.5, 1.2  # Mutation factor range
        CR_min, CR_max = 0.6, 1.0  # Crossover probability range

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation: DE/rand/1 strategy with adaptive factor
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover: Binomial
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < self.CR else population[i][j]
                        for j in range(self.dimension)
                    ]
                )
                trial_fitness = func(trial)
                evaluations += 1

                # Selection: Greedy
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            # Adaptation of F and CR
            mean_fitness = np.mean(fitness)
            if best_fitness < mean_fitness:
                self.F = min(F_max, self.F + 0.03 * (mean_fitness - best_fitness))
                self.CR = max(CR_min, self.CR - 0.01 * (mean_fitness - best_fitness))
            else:
                self.F = max(F_min, self.F - 0.01)
                self.CR = min(CR_max, self.CR + 0.01)

        return best_fitness, best_individual
