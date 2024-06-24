import numpy as np


class RADE:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size

    def __call__(self, func):
        # Initialization of population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        # Differential Evolution parameters
        mutation_factor = 0.8
        crossover_probability = 0.9
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation strategy: "rand/1/bin"
                indices = [index for index in range(self.population_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < crossover_probability else population[i][j]
                        for j in range(self.dimension)
                    ]
                )
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            # Adaptive strategy adjustment
            mutation_factor = max(0.5, min(1.0, mutation_factor + 0.02 * (best_fitness - np.mean(fitness))))
            crossover_probability = max(
                0.5, min(1.0, crossover_probability - 0.02 * (best_fitness - np.mean(fitness)))
            )

        return best_fitness, best_individual
