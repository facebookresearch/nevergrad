import numpy as np


class EnhancedSequentialQuadraticAdaptiveEvolutionStrategy:
    def __init__(self, budget, dimension=5, population_size=50, F_init=0.5, CR_init=0.8, alpha=0.1):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.alpha = alpha  # Introducing a learning rate for F and CR adjustments
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.best_global_fitness = np.inf
        self.best_global_solution = None

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

    def adjust_parameters(self, iteration):
        # Gradually decrease F and increase CR based on progression and best fitness found
        scale = iteration / self.budget
        self.F = max(0.1, self.F - self.alpha * scale * (self.F - 0.1))
        self.CR = min(1.0, self.CR + self.alpha * scale * (1 - self.CR))

    def update_global_best(self, population, fitnesses):
        local_best_index = np.argmin(fitnesses)
        local_best_fitness = fitnesses[local_best_index]
        if local_best_fitness < self.best_global_fitness:
            self.best_global_fitness = local_best_fitness
            self.best_global_solution = population[local_best_index].copy()

    def __call__(self, func):
        population = self.initialize_population()
        self.fitness = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0

        while evaluations < self.budget:
            self.adjust_parameters(iteration)
            self.update_global_best(population, self.fitness)

            for i in range(self.pop_size):
                mutant = self.mutate(population, i)
                trial = self.crossover(population[i], mutant)
                population[i], self.fitness[i] = self.select(population[i], trial, func)
                evaluations += 1
                if evaluations >= self.budget:
                    break
            iteration += 1

        return self.best_global_fitness, self.best_global_solution
