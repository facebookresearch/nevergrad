import numpy as np


class EnhancedPhaseTransitionMemoryStrategyV82:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        F_init=0.5,
        CR_init=0.9,
        memory_size=10,
        phase_transition=0.5,
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.memory_size = memory_size
        self.phase_transition = phase_transition
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, evaluations):
        phase = 1 if evaluations < self.budget * self.phase_transition else 2
        idxs = np.random.choice(self.pop_size, 3, replace=False)
        x1, x2, x3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        if phase == 1:
            mutant = x1 + self.F * (x2 - x3)
        else:
            memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
            mutant = x1 + self.F * (x2 - x3) + 0.1 * memory_effect
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
                self.memory.pop(0)
                self.memory.append(trial - target)
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
