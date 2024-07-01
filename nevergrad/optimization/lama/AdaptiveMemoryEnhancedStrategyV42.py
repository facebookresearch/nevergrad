import numpy as np


class AdaptiveMemoryEnhancedStrategyV42:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)

        # Introduce memory-guided mutation
        memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
        mutant = population[a] + self.F * (population[b] - population[c]) + memory_effect
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            self.memory.append(trial - target)
            if len(self.memory) > 10:  # Limit memory size
                self.memory.pop(0)
            return trial, f_trial
        else:
            return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        # Adjust parameters using a sigmoid-based adaptive method
        scale = iteration / total_iterations
        self.F = np.clip(0.5 + 0.5 * np.sin(2 * np.pi * scale), 0.1, 1)
        self.CR = np.clip(0.5 + 0.5 * np.cos(2 * np.pi * scale), 0.1, 1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            self.adjust_parameters(iteration, self.budget)

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
            iteration += 1

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
