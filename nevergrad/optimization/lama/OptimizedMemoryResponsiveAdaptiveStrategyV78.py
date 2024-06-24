import numpy as np


class OptimizedMemoryResponsiveAdaptiveStrategyV78:
    def __init__(self, budget, dimension=5, population_size=100, memory_size=10):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []
        self.F_base = 0.5
        self.CR_base = 0.9

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        a, b, c = np.random.choice(size, 3, replace=False)
        mutation_factor = self.F_base + 0.5 * np.sin(
            2 * np.pi * (1 - (index / self.budget))
        )  # Dynamic F based on progress
        mutant = population[a] + mutation_factor * (population[b] - population[c])
        if self.memory:
            memory_effect = np.mean(self.memory, axis=0)
            mutant += 0.1 * memory_effect  # Reduced memory influence
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_rate = self.CR_base + 0.5 * np.cos(
            2 * np.pi * (1 - (len(self.memory) / self.memory_size))
        )  # Dynamic CR based on memory usage
        crossover_mask = np.random.rand(self.dimension) < crossover_rate
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            if len(self.memory) < self.memory_size:
                self.memory.append(trial - target)
            else:
                # Replace older memories based on a probability related to improvement magnitude
                if np.random.rand() < (f_target - f_trial) / f_target:
                    self.memory[np.random.randint(self.memory_size)] = trial - target
            return trial, f_trial
        return target, f_target

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, evaluations)
                trial = self.crossover(population[i], mutant)
                population[i], fitnesses[i] = self.select(population[i], trial, func)
                if fitnesses[i] < fitnesses[best_idx]:
                    best_idx = i
                evaluations += 1
                if evaluations >= self.budget:
                    break

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
