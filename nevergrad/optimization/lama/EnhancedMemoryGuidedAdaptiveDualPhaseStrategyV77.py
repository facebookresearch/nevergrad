import numpy as np


class EnhancedMemoryGuidedAdaptiveDualPhaseStrategyV77:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, memory_size=10):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F_init = F_init
        self.CR_init = CR_init
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx):
        size = len(population)
        idxs = np.random.choice(size, 3, replace=False)
        a, b, c = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        if self.memory:
            memory_effect = np.mean(self.memory, axis=0)
            mutant = a + self.F_init * (best_idx - b) + memory_effect
        else:
            mutant = a + self.F_init * (b - c)
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR_init
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            self.update_memory(trial - target)
            return trial, f_trial
        return target, f_target

    def update_memory(self, diff):
        if len(self.memory) < self.memory_size:
            self.memory.append(diff)
        else:
            # Replace an old memory with new one probabilistically
            replace_idx = np.random.randint(0, self.memory_size)
            self.memory[replace_idx] = diff

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx)
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
