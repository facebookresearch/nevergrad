import numpy as np


class EnhancedDynamicEscapeStrategyV32:
    def __init__(
        self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, escape_threshold=0.1
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init  # Initial mutation factor
        self.CR = CR_init  # Initial crossover rate
        self.escape_threshold = escape_threshold  # Threshold to trigger an escape mechanism
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index, iteration, func):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        # Dynamic mutation factor adjustment based on iteration progress
        F_dynamic = self.F * (1 + 0.5 * np.sin(2 * np.pi * iteration / self.budget))
        mutant = population[best_idx] + F_dynamic * (
            population[a] - population[b] + population[c] - population[best_idx]
        )
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant, iteration):
        # Dynamic crossover rate adjustment based on iteration progress
        CR_dynamic = self.CR * (1 + 0.5 * np.cos(2 * np.pi * iteration / self.budget))
        crossover_mask = np.random.rand(self.dimension) < CR_dynamic
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            return trial, f_trial
        else:
            return target, f_target

    def escape_mechanism(self, population, fitnesses, func):
        # Trigger an escape mechanism if population converges
        if np.std(fitnesses) < self.escape_threshold:
            # Reinitialize a portion of the population
            num_escape = self.pop_size // 5
            escape_indices = np.random.choice(range(self.pop_size), num_escape, replace=False)
            for index in escape_indices:
                population[index] = np.random.uniform(self.lower_bounds, self.upper_bounds, self.dimension)
                fitnesses[index] = func(population[index])
        return population, fitnesses

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i, evaluations, func)
                trial = self.crossover(population[i], mutant, evaluations)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i], fitnesses[i] = trial, trial_fitness
                    if trial_fitness < fitnesses[best_idx]:
                        best_idx = i

            population, fitnesses = self.escape_mechanism(population, fitnesses, func)

            if evaluations >= self.budget:
                break

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
