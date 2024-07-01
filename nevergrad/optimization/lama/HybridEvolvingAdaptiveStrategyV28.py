import numpy as np


class HybridEvolvingAdaptiveStrategyV28:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9, switch_ratio=0.5):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init  # Mutation factor start value
        self.CR = CR_init  # Crossover rate start value
        self.switch_ratio = switch_ratio
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, idx, phase):
        size = len(population)
        idxs = [i for i in range(size) if i != idx]
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
        if f_trial < f_target:
            return trial, f_trial
        return target, f_target

    def adjust_parameters(self, iteration, improvements):
        # Adjust mutation factor and crossover rate dynamically based on historical improvements
        self.F = np.clip(self.F + 0.01 * improvements, 0.1, 1)
        self.CR = np.clip(self.CR - 0.01 * improvements, 0.1, 1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        recent_improvements = 0

        while evaluations < self.budget:
            phase = 1 if evaluations < self.budget * self.switch_ratio else 2
            for i in range(self.pop_size):
                mutant = self.mutate(population, best_idx, i, phase)
                trial = self.crossover(population[i], mutant)
                trial, trial_fitness = self.select(population[i], trial, func)
                evaluations += 1

                if trial_fitness < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_idx = i
                        best_fitness = trial_fitness
                        recent_improvements += 1

                if evaluations >= self.budget:
                    break
            self.adjust_parameters(evaluations, recent_improvements)
            recent_improvements = 0  # Reset improvements counter after each parameter adjustment

        return best_fitness, population[best_idx]
