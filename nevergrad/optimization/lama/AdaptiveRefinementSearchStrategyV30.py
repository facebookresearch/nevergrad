import numpy as np


class AdaptiveRefinementSearchStrategyV30:
    def __init__(
        self, budget, dimension=5, population_size=100, F_max=0.9, F_min=0.1, CR_max=0.9, CR_min=0.4
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F_max = F_max
        self.F_min = F_min
        self.CR_max = CR_max
        self.CR_min = CR_min
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        scale = np.random.uniform(self.F_min, self.F_max)
        mutant = population[best_idx] + scale * (
            population[a] - population[b] + population[c] - population[best_idx]
        )
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        CR = np.random.uniform(self.CR_min, self.CR_max)
        crossover_mask = np.random.rand(self.dimension) < CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target, trial, func, target_fitness, trial_fitness):
        if trial_fitness < target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1
                population[i], fitnesses[i] = self.select(
                    population[i], trial, func, fitnesses[i], trial_fitness
                )

                if fitnesses[i] < fitnesses[best_idx]:
                    best_idx = i

                if evaluations >= self.budget:
                    break

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
