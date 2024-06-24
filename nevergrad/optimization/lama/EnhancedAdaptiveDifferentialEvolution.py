import numpy as np


class EnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, dimension=5, population_size=50, F=0.8, CR=0.9, adaptive=True):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F
        self.CR = CR
        self.adaptive = adaptive
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            return trial, f_trial
        else:
            return target, f_target

    def adjust_parameters(self):
        self.F = np.clip(np.random.normal(self.F, 0.1), 0.5, 1.0)
        self.CR = np.clip(np.random.normal(self.CR, 0.1), 0.8, 1.0)

    def __call__(self, func):
        population = self.initialize_population()
        self.fitness = np.array([func(ind) for ind in population])
        evaluations = len(population)

        while evaluations < self.budget:
            if self.adaptive:
                self.adjust_parameters()

            for i in range(self.pop_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                population[i], self.fitness[i] = self.select(population[i], trial, func)
                evaluations += 1
                if evaluations >= self.budget:
                    break

        f_opt = np.min(self.fitness)
        x_opt = population[np.argmin(self.fitness)]
        return f_opt, x_opt
