import numpy as np


class AdvancedAdaptiveMemoryEnhancedStrategyV73:
    def __init__(self, budget, dimension=5, population_size=50, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []
        self.memory_size = 20  # Enhanced memory management

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        candidates = list(range(size))
        candidates.remove(index)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutation_factor = self.F * np.tanh(4 * (1 - (index / self.pop_size)))  # Adaptive mutation factor
        mutant = (
            population[a]
            + mutation_factor * (population[b] - population[c])
            + 0.1 * (population[best_idx] - population[index])
        )
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            if len(self.memory) < self.memory_size:
                self.memory.append(trial - target)
            else:
                self.memory[np.random.randint(len(self.memory))] = (
                    trial - target
                )  # Random replacement strategy
            return trial, f_trial
        return target, f_target

    def adaptive_memory_effect(self):
        if self.memory:
            return np.mean(self.memory, axis=0)
        return np.zeros(self.dimension)

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
