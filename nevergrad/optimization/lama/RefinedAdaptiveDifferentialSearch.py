import numpy as np


class RefinedAdaptiveDifferentialSearch:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.adaptive_factor = 0.05  # Smaller adaptive factor to fine-tune exploration

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, population, idx):
        idxs = [i for i in range(self.population_size) if i != idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = np.clip(
            population[a] + self.mutation_factor * (population[b] - population[c]),
            self.bounds[0],
            self.bounds[1],
        )
        return mutant

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.crossover_rate
        return np.where(crossover_mask, mutant, target)

    def select(self, current, candidate, func):
        if func(candidate) < func(current):
            return candidate
        else:
            return current

    def optimize(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                population[i] = self.select(population[i], trial, func)

                # Update best solution found so far
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    if trial_fitness < fitness[best_idx]:
                        best_individual = trial
                        best_idx = i

                evaluations += 1
                if evaluations >= self.budget:
                    break

        return fitness[best_idx], best_individual

    def __call__(self, func):
        return self.optimize(func)
