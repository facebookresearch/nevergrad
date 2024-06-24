import numpy as np


class EnhancedAdaptiveGranularStrategyV26:
    def __init__(
        self, budget, dimension=5, population_size=100, F_min=0.4, F_max=0.9, CR_min=0.5, CR_max=0.9
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

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (
            population[b] - population[c] + 0.5 * (population[best_idx] - population[c])
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
        progress = iteration / total_iterations
        self.F = np.clip(
            self.F_min + (self.F_max - self.F_min) * np.sin(np.pi * progress), self.F_min, self.F_max
        )
        self.CR = np.clip(
            self.CR_min + (self.CR_max - self.CR_min) * np.cos(np.pi * progress), self.CR_min, self.CR_max
        )

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = self.pop_size
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            self.adapt_parameters(evaluations, self.budget)

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

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
