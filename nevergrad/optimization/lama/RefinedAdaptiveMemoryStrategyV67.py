import numpy as np


class RefinedAdaptiveMemoryStrategyV67:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        F_init=0.5,
        CR_init=0.9,
        switch_ratio=0.5,
        memory_size=20,
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init  # Mutation factor
        self.CR = CR_init  # Crossover rate
        self.switch_ratio = switch_ratio
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index, phase):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        if phase == 1:
            mutant = population[best_idx] + self.F * (population[a] - population[b])
        else:
            # Use memory to guide mutation in phase 2
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
            if len(self.memory) < self.memory_size:
                self.memory.append(trial - target)
            else:
                self.memory.pop(0)
                self.memory.append(trial - target)
            return trial, f_trial
        else:
            return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        scale = iteration / total_iterations
        # Gradual and non-linear parameter adaptation
        self.F = 0.5 + 0.5 * np.sin(np.pi * scale)
        self.CR = 0.5 + 0.5 * np.cos(np.pi * scale)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        total_iterations = self.budget // self.pop_size
        switch_point = int(self.switch_ratio * total_iterations)

        for iteration in range(total_iterations):
            phase = 1 if iteration < switch_point else 2
            self.adjust_parameters(iteration, total_iterations)

            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i, phase)
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
