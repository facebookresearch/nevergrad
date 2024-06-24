import numpy as np


class DynamicMemoryAdaptiveConvergenceStrategyV76:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx):
        size = len(population)
        a, b, c, d = np.random.choice(size, 4, replace=False)
        memory_effect = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
        mutant = (
            population[a] + self.F * (population[b] - population[c]) + 0.1 * memory_effect
        )  # Memory-influenced mutation
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target or np.random.rand() < 0.05:  # Probabilistic acceptance
            if len(self.memory) < 10:
                self.memory.append(trial - target)
            else:
                self.memory[np.random.randint(len(self.memory))] = trial - target
            return trial, f_trial
        return target, f_target

    def adapt_parameters(self, success_rate):
        if success_rate < 0.1:
            self.F = max(0.1, self.F - 0.1)
            self.CR = min(0.9, self.CR + 0.1)
        elif success_rate > 0.2:
            self.F = min(1.0, self.F + 0.1)
            self.CR = max(0.1, self.CR - 0.1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        last_best = fitnesses[best_idx]
        successes = 0

        while evaluations < self.budget:
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx)
                trial = self.crossover(population[i], mutant)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    if trial_fitness < fitnesses[best_idx]:
                        best_idx = i
                        successes += 1

                if evaluations >= self.budget:
                    break

            current_best = fitnesses[best_idx]
            if current_best < last_best:
                success_rate = successes / self.pop_size
                self.adapt_parameters(success_rate)
                last_best = current_best
                successes = 0

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
