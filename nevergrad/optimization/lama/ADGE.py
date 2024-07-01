import numpy as np


class ADGE:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, func):
        population_size = 100
        mutation_factor = 0.8
        recombination_crossover = 0.9
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        evaluations = population_size
        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < recombination_crossover else population[i][j]
                        for j in range(self.dimension)
                    ]
                )
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            # Adapt mutation factor and crossover based on progress
            if evaluations % 100 == 0:
                mutation_factor = max(0.5, mutation_factor * 0.95)
                recombination_crossover = min(1.0, recombination_crossover + 0.05)

        return best_fitness, best_individual
