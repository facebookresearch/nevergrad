import numpy as np


class OptimizedMemoryGuidedAdaptiveStrategyV81:
    def __init__(
        self, budget, dimension=5, population_size=100, F_base=0.5, CR_base=0.9, memory_size=15, adaptive=True
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.memory_size = memory_size
        self.adaptive = adaptive
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, phase):
        size = len(population)
        indices = np.random.choice(size, 3, replace=False)
        a, b, c = indices[0], indices[1], indices[2]
        if phase == 1:  # Exploration
            mutant = population[a] + self.F_base * (population[b] - population[c])
        else:  # Exploitation
            memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
            mutant = population[a] + self.F_base * (population[b] - population[c]) + 0.1 * memory_effect
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR_base
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            if len(self.memory) < self.memory_size:
                self.memory.append(trial - target)
            else:
                self.memory.pop(0)
                self.memory.append(trial - target)
            return trial, f_trial
        return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        if self.adaptive:
            self.F_base = 0.5 + 0.4 * np.sin(np.pi * iteration / total_iterations)
            self.CR_base = 0.6 + 0.3 * np.cos(np.pi * iteration / total_iterations)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0
        best_idx = np.argmin(fitnesses)
        while evaluations < self.budget:
            self.adjust_parameters(iteration, self.budget)

            for i in range(self.pop_size):
                phase = 1 if evaluations < self.budget / 2 else 2
                mutant = self.mutate(population, best_idx, phase)
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
