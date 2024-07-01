import numpy as np


class HyperAdaptiveMemoryGuidedStrategyV74:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.6, CR_init=0.9, memory_size=20):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.memory = []
        self.memory_size = memory_size

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        candidates = list(range(size))
        candidates.remove(index)
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant_base = population[a] + self.F * (population[b] - population[c])
        memory_contribution = np.mean(self.memory, axis=0) if self.memory else np.zeros(self.dimension)
        mutant = mutant_base + 0.5 * memory_contribution  # Memory influenced mutation
        return np.clip(mutant, self.lower_bounds, self.upper_bounds)

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.CR
        return np.where(crossover_mask, mutant, target)

    def select(self, target, trial, func):
        f_target = func(target)
        f_trial = func(trial)
        if f_trial <= f_target:
            if len(self.memory) < self.memory_size:
                self.memory.append(trial - target)
            elif np.random.rand() < 0.15:  # Probabilistic replacement
                self.memory[np.random.randint(len(self.memory))] = trial - target
            return trial, f_trial
        return target, f_target

    def adapt_parameters(self, current_eval, total_budget):
        progress = current_eval / total_budget
        self.F = 0.5 + 0.5 * np.sin(np.pi * progress)  # Dynamic adaptation of F
        self.CR = 0.5 + 0.5 * np.cos(np.pi * progress)  # Dynamic adaptation of CR

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
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
