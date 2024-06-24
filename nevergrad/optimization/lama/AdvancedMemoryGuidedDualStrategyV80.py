import numpy as np


class AdvancedMemoryGuidedDualStrategyV80:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, switch_ratio=0.5):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.switch_ratio = switch_ratio
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, phase):
        size = len(population)
        indices = np.random.choice(size, 3, replace=False)
        a, b, c = indices[0], indices[1], indices[2]
        # Leveraging memory for mutation depending on the optimization phase
        if phase == 1:
            # Exploration phase
            mutant = population[best_idx] + self.F * (population[b] - population[c])
        else:
            # Exploitation phase
            memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
            mutant = population[a] + self.F * (population[b] - population[c]) + 0.1 * memory_effect
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            if len(self.memory) < 10:  # Limit memory size
                self.memory.append(trial - target)
            else:
                self.memory.pop(0)
                self.memory.append(trial - target)
            return trial, f_trial
        return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        # Dynamic adjustment of F and CR
        self.F = 0.5 + 0.5 * np.sin(np.pi * (iteration / total_iterations))
        self.CR = 0.9 - 0.4 * np.cos(np.pi * (iteration / total_iterations))

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0
        best_idx = np.argmin(fitnesses)
        switch_point = int(self.switch_ratio * self.budget)

        while evaluations < self.budget:
            phase = 1 if evaluations < switch_point else 2
            self.adjust_parameters(iteration, switch_point if phase == 1 else self.budget - switch_point)

            for i in range(self.pop_size):
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
