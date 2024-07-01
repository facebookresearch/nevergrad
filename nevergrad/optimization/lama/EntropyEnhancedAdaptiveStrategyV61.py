import numpy as np


class EntropyEnhancedAdaptiveStrategyV61:
    def __init__(self, budget, dimension=5, population_size=100, F_init=0.5, CR_init=0.9):
        self.budget = budget
        self.dimension = dimension
        self.pop_size = population_size
        self.F = F_init
        self.CR = CR_init
        self.lower_bounds = -5.0 * np.ones(self.dimension)
        self.upper_bounds = 5.0 * np.ones(self.dimension)
        self.entropy_threshold = 0.05  # Threshold to adjust mutation rates

    def initialize_population(self):
        return np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.dimension))

    def mutate(self, population, best_idx, index):
        size = len(population)
        idxs = [idx for idx in range(size) if idx != index]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        mutant = population[a] + self.F * (population[b] - population[c])
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

    def calculate_population_entropy(self, population):
        norm_pop = (population - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        digitized = np.digitize(norm_pop, bins=np.linspace(0, 1, 10))
        hist = [np.histogram(digitized[:, i], bins=10, range=(1, 10))[0] for i in range(self.dimension)]
        hist = np.array(hist) + 1  # Avoid log(0)
        probs = hist / np.sum(hist, axis=1, keepdims=True)
        entropy = -np.sum(probs * np.log(probs), axis=1)
        return np.mean(entropy)

    def adjust_parameters(self, population):
        entropy = self.calculate_population_entropy(population)
        if entropy < self.entropy_threshold:
            self.F = np.clip(self.F * 0.9, 0.1, 1)  # Decrease mutation rate if diversity is low
            self.CR = np.clip(self.CR * 1.1, 0.1, 1)  # Increase crossover rate to introduce more diversity
        else:
            self.F = np.clip(self.F * 1.1, 0.1, 1)
            self.CR = np.clip(self.CR * 0.9, 0.1, 1)

    def __call__(self, func):
        population = self.initialize_population()
        fitnesses = np.array([func(ind) for ind in population])
        evaluations = len(population)
        best_idx = np.argmin(fitnesses)

        while evaluations < self.budget:
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

            self.adjust_parameters(population)

        best_fitness = fitnesses[best_idx]
        best_solution = population[best_idx]
        return best_fitness, best_solution
