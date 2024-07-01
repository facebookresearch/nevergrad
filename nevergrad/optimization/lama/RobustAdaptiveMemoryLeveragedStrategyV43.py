import numpy as np


class RobustAdaptiveMemoryLeveragedStrategyV43:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, switch_ratio=0.5):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.switch_ratio = switch_ratio
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []
        self.phase = 1

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        if self.phase == 1:
            mutant = population[best_idx] + self.F * (population[a] - population[b])
        else:
            # Enhance mutation strategy by adapting based on memory depth
            memory_factor = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
            mutant = population[a] + self.F * (population[b] - population[c]) + 0.5 * memory_factor
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            self.memory.append(trial - target)
            if len(self.memory) > 20:  # Adjusted memory size
                self.memory.pop(0)
            return trial, f_trial
        else:
            return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        # Adaptive parameter tuning based on sigmoid-based dynamic adjustment
        scale = iteration / total_iterations
        sigmoid_factor = 1 / (1 + np.exp(-12 * (scale - 0.5)))
        self.F = np.clip(0.6 * sigmoid_factor, 0.1, 1)
        self.CR = np.clip(0.8 * sigmoid_factor, 0.1, 1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        iteration = 0
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
            self.adjust_parameters(iteration, self.budget)
            self.phase = 1 if evaluations < self.budget * self.switch_ratio else 2

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
            iteration += 1

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
