import numpy as np


class RefinedIntelligentEvolvingAdaptiveStrategyV35:
    def __init__(self, budget, dimension=5, population_size=50, F_init=0.8, CR_init=0.9):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.history = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b = np.random.choice(idxs, 2, replace=False)
        # Using a simpler, more stable mutation strategy
        mutant = population[a] + self.F * (population[b] - population[best_idx])
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            return trial, f_trial
        else:
            return target, f_target

    def adjust_parameters(self, iteration):
        # Simplifying parameter adjustment to focus on decaying F and stable CR
        self.F = 0.8 * np.exp(-2 * iteration / self.budget)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        iteration = 0
        while evaluations < self.budget:
            self.adjust_parameters(iteration)

            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i)
                trial = self.crossover(population[i], mutant)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i], fitnesses[i] = trial, trial_fitness
                    if trial_fitness < fitnesses[best_idx]:
                        best_idx = i
                        self.history.append((trial, trial_fitness))

                if evaluations >= self.budget:
                    break
            iteration += 1

        # Optional: Retrieve the best found solution in history instead of the current population
        if self.history:
            best_solution, best_fitness = min(self.history, key=lambda x: x[1])
        else:
            best_fitness = fitnesses[best_idx]
            best_solution = population[best_idx]

        return best_fitness, best_solution
