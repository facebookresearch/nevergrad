import numpy as np


class EnhancedStrategicMemoryAdaptiveStrategyV44:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, memory_size=10):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init  # Mutation factor
        self.CR = CR_init  # Crossover probability
        self.memory_size = memory_size
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
        if len(self.memory) > 0:
            memory_factor = self.memory[np.random.randint(len(self.memory))]
            mutant += memory_factor  # Integrating memory effect into mutation
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial < f_target:
            self.memory.append(trial - target)
            if len(self.memory) > self.memory_size:
                self.memory.pop(0)
            return trial, f_trial
        else:
            return target, f_target

    def adjust_parameters(self, iteration, total_iterations):
        sigmoid_adjustment = 1 / (1 + np.exp(-10 * (iteration / total_iterations - 0.5)))
        self.F = np.clip(0.5 + 0.4 * sigmoid_adjustment, 0.1, 0.9)
        self.CR = np.clip(0.5 + 0.4 * np.sin(sigmoid_adjustment), 0.1, 0.9)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        iteration = 0

        while evaluations < self.budget:
            self.adjust_parameters(iteration, self.budget)
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
