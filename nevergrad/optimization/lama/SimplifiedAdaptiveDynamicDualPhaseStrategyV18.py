import numpy as np


class SimplifiedAdaptiveDynamicDualPhaseStrategyV18:
    def __init__(
        self, budget, dimension=5, population_size=100, F_min=0.5, F_max=0.8, CR_min=0.8, CR_max=0.9
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index, phase):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        if phase == 2:
            d, e = np.random.choice(idxs, 2, replace=False)
            mutant += 0.5 * self.F * (population[d] - population[e])
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        return (trial, f_trial) if f_trial < f_target else (target, f_target)

    def adjust_parameters(self, iteration, total_iterations):
        # Using a linear schedule for parameter adaptation
        t = iteration / total_iterations
        self.F = self.F_min + t * (self.F_max - self.F_min)
        self.CR = self.CR_max - t * (self.CR_max - self.CR_min)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            self.adjust_parameters(iteration, self.budget / self.pop_size)

            for i in range(self.pop_size):
                phase = 2 if iteration > (self.budget / self.pop_size) / 2 else 1
                mutant = self.mutate(population, best_idx, i, phase)
                trial = self.crossover(population[i], mutant)
                population[i], fitnesses[i] = self.select(population[i], trial, func)
                evaluations += 1
                if fitnesses[i] < fitnesses[best_idx]:
                    best_idx = i
                if evaluations >= self.budget:
                    break
            iteration += 1

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
