import numpy as np


class OptimizedAdaptiveDynamicStrategyV34:
    def __init__(self, budget, dimension=5, population_size=80, F_init=0.6, CR_init=0.85):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index, iteration):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        # Incorporate adaptive mutation factor adjusted by a logarithmic function
        F_dynamic = self.F / np.log(iteration + 2)
        mutant = (
            population[best_idx]
            + F_dynamic * (population[a] - population[b])
            + 0.1 * (population[c] - population[best_idx])
        )
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant, iteration):
        # Implement adaptive crossover probability, varying over the course of iterations
        CR_dynamic = self.CR * (0.5 + 0.5 * np.sin(2 * np.pi * iteration / self.budget))
        crossover_mask = np.random.rand(self.dimension) < CR_dynamic
        trial = np.where(crossover_mask, mutant, target)
        return trial

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            return trial, f_trial
        else:
            return target, f_target

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        iteration = 0
        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i, iteration)
                trial = self.crossover(population[i], mutant, iteration)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i], fitnesses[i] = trial, trial_fitness
                    if trial_fitness < fitnesses[best_idx]:
                        best_idx = i
            iteration += 1

            if evaluations >= self.budget:
                break

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
