import numpy as np


class EnhancedAdaptiveMemoryStrategyV79:
    def __init__(self, budget, dimension=5, population_size=50, memory_size=10):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, phase):
        size = len(population)
        a, b, c = np.random.choice(size, 3, replace=False)
        F = 0.5 + 0.4 * np.sin(np.pi * phase)  # Dynamic mutation factor based on phase
        if phase < 0.5:
            # Exploration phase
            mutant = population[a] + F * (population[b] - population[c])
        else:
            # Exploitation phase using memory
            memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
            mutant = population[best_idx] + F * (population[b] - population[c]) + 0.1 * memory_effect
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        CR = 0.9 - 0.4 * np.cos(np.pi * len(self.memory) / self.memory_size)  # Dynamic crossover rate
        crossover_mask = np.random.rand(self.dimension) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            if len(self.memory) < self.memory_size:
                self.memory.append(trial - target)
            else:
                self.memory[np.random.randint(self.memory_size)] = trial - target
            return trial, f_trial
        return target, f_target

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        phase = 0

        while evaluations < self.budget:
            phase_progress = evaluations / self.budget
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, phase_progress)
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
