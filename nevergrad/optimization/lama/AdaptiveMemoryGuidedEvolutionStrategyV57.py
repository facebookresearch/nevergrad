import numpy as np


class AdaptiveMemoryGuidedEvolutionStrategyV57:
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
        self.success_memory = []
        self.failure_memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        memory_effect = (
            np.mean(self.success_memory, axis=0) if self.success_memory else np.zeros(self.dimension)
        )
        failure_effect = (
            np.mean(self.failure_memory, axis=0) if self.failure_memory else np.zeros(self.dimension)
        )
        F = np.clip(
            self.F + 0.1 * np.sin(len(self.success_memory)), 0.1, 1.0
        )  # Adaptive F based on memory size
        mutant = population[a] + F * (population[b] - population[c] + memory_effect - 0.1 * failure_effect)
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            self.success_memory.append(trial - target)
            if len(self.success_memory) > self.memory_size:
                self.success_memory.pop(0)
            return trial, f_trial
        else:
            self.failure_memory.append(trial - target)
            if len(self.failure_memory) > self.memory_size:
                self.failure_memory.pop(0)
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
                population[i], fitnesses[i] = self.select(population[i], trial, func)
                evaluations += 1

                if fitnesses[i] < fitnesses[best_idx]:
                    best_idx = i

                if evaluations >= self.budget:
                    break

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
