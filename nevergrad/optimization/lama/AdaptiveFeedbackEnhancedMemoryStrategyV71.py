import numpy as np


class AdaptiveFeedbackEnhancedMemoryStrategyV71:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=50,
        F_init=0.5,
        CR_init=0.9,
        switch_ratio=0.5,
        memory_size=20,
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init  # Mutation factor
        self.CR = CR_init  # Crossover factor
        self.switch_ratio = switch_ratio
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []
        self.best_f = float("inf")
        self.best_solution = None

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index, phase):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        if phase == 1:
            mutant = population[best_idx] + self.F * (population[a] - population[b])
        else:
            memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
            mutant = population[c] + self.F * (population[a] - population[b]) + memory_effect
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
        else:
            return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        # Dynamic parameter adjustment based on feedback and performance
        self.F = 0.5 + (0.5 * np.sin(np.pi * (iteration / total_iterations)))
        self.CR = 0.9 - (0.4 * np.cos(np.pi * (iteration / total_iterations)))

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        self.best_f = fitnesses[best_idx]
        self.best_solution = population[best_idx]
        total_iterations = self.budget // self.pop_size

        for iteration in range(total_iterations):
            phase = 1 if iteration < total_iterations * self.switch_ratio else 2
            self.adjust_parameters(iteration, total_iterations)

            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i, phase)
                trial = self.crossover(population[i], mutant)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    if trial_fitness < self.best_f:
                        best_idx = i
                        self.best_f = trial_fitness
                        self.best_solution = trial

                if evaluations >= self.budget:
                    break

        return self.best_f, self.best_solution
