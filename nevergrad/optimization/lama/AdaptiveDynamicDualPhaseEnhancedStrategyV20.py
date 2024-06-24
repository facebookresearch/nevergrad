import numpy as np


class AdaptiveDynamicDualPhaseEnhancedStrategyV20:
    def __init__(self, budget, dimension=5, population_size=100, F_base=0.5, CR_base=0.9, switch_ratio=0.5):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F_base = F_base
        self.CR_base = CR_base
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
            # Enhanced mutation strategy: introduces additional diversity by considering more vectors
            d, e = np.random.choice(idxs, 2, replace=False)
            mutant = population[a] + self.F * (
                population[b] - population[c] + 0.5 * (population[d] - population[e])
            )
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        # Adaptive crossover rate according to the performance of individuals
        CR = self.CR * (1 + (np.std(target) / (np.mean(target) + 1e-10)))
        crossover_mask = np.random.rand(self.dimension) < CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        return (trial, f_trial) if f_trial < f_target else (target, f_target)

    def adjust_parameters(self, iteration, total_iterations):
        # Dynamic adjustment of parameters using a sigmoid function
        scale = iteration / total_iterations
        logistic = 1 / (1 + np.exp(-10 * (scale - 0.5)))
        self.F = np.clip(self.F_base + 0.5 * np.sin(2 * np.pi * scale), 0.1, 1)
        self.CR = np.clip(self.CR_base + 0.5 * np.cos(2 * np.pi * scale), 0.1, 1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0
        best_idx = np.argmin(fitnesses)
        switch_point = int(self.switch_ratio * self.budget)

        while evaluations < self.budget:
            self.adjust_parameters(
                iteration, switch_point if evaluations < switch_point else self.budget - switch_point
            )
            phase = 1 if evaluations < switch_point else 2

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
            iteration += 1

        return fitnesses[best_idx], population[best_idx]
