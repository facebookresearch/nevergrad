import numpy as np


class AdvancedAdaptiveMemoryEnhancedStrategyV56:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, memory_size=20):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        # Adaptive mutation factor based on progress
        current_progress = len(self.memory) / self.memory_size
        F = self.F * (1 - current_progress) + 0.1 * current_progress
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + F * (population[b] - population[c])

        # Memory-guided mutation
        if self.memory:
            memory_effect = np.mean(self.memory, axis=0)
            mutant += 0.1 * memory_effect

        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        # Adaptive crossover rate
        crossover_rate = self.CR * (1 - (len(self.memory) / self.memory_size))
        crossover_mask = np.random.rand(self.dimension) < crossover_rate
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            self.memory.append(trial - target)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            return trial, f_trial
        else:
            return target, f_target

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i)
                trial = self.crossover(population[i], mutant)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    if trial_fitness < fitnesses[best_idx]:
                        best_idx = i

                if evaluations >= self.budget:
                    break

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
