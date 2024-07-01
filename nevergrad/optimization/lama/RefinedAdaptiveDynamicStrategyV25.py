import numpy as np


class RefinedAdaptiveDynamicStrategyV25:
    def __init__(
        self, budget, dimension=5, population_size=100, initial_F=0.5, initial_CR=0.9, switch_ratio=0.5
    ):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.initial_F = initial_F
        self.initial_CR = initial_CR
        self.switch_ratio = switch_ratio
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index, phase):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        if phase == 1:
            mutant = population[best_idx] + self.F * (population[a] - population[b])
        else:
            d, e = np.random.choice(idxs, 2, replace=False)
            mutant = population[a] + self.F * (
                population[b] - population[c] + 0.5 * (population[d] - population[e])
            )
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        return (trial, f_trial) if f_trial < f_target else (target, f_target)

    def adapt_parameters(self, iteration, total_iterations):
        scale = iteration / total_iterations
        sigmoid = 1 / (1 + np.exp(-12 * (scale - 0.5)))
        self.F = np.clip(self.initial_F + 0.4 * np.sin(np.pi * sigmoid), 0.1, 1)
        self.CR = np.clip(self.initial_CR + 0.4 * np.cos(np.pi * sigmoid), 0.1, 1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        switch_point = int(self.switch_ratio * self.budget)

        while evaluations < self.budget:
            phase = 1 if evaluations < switch_point else 2
            self.adapt_parameters(evaluations, self.budget)

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
