import numpy as np


class DEAMC_DSR:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.F_base = 0.8  # Base mutation scaling factor
        self.CR_base = 0.9  # Base crossover probability
        self.stagnation_threshold = 30  # Threshold for stagnation detection
        self.no_improvement_intervals = 0

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, idx, population):
        indices = np.delete(np.arange(self.population_size), idx)
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F_base * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.CR_base
        return np.where(cross_points, mutant, target)

    def select(self, target, trial, f_target, f_trial):
        return trial if f_trial < f_target else target

    def local_search(self, best_individual, func):
        step_size = 0.1
        for _ in range(10):  # Perform 10 steps of local search
            neighbor = best_individual + np.random.uniform(-step_size, step_size, self.dimension)
            neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
            if func(neighbor) < func(best_individual):
                best_individual = neighbor
        return best_individual

    def adapt_and_refine(self, successes, trials, best_idx, population, fitness, func):
        if successes / trials < 0.1:  # High stagnation detected
            self.no_improvement_intervals += 1
        else:
            self.no_improvement_intervals = 0

        self.F_base = max(0.1, self.F_base * (0.95 if successes / trials < 0.2 else 1.05))
        self.CR_base = 0.1 + 0.8 * successes / trials  # Adaptive CR within [0.1, 0.9]

        # Trigger local search on stagnation
        if self.no_improvement_intervals >= self.stagnation_threshold:
            population[best_idx] = self.local_search(population[best_idx], func)
            fitness[best_idx] = func(population[best_idx])
            self.no_improvement_intervals = 0  # reset after local search

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)

        while evaluations < self.budget:
            new_population = np.copy(population)
            successful_trials = 0

            for i in range(self.population_size):
                mutant = self.mutation(i, population)
                trial = self.crossover(population[i], mutant)
                f_trial = func(trial)
                f_target = fitness[i]

                if f_trial < f_target:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    successful_trials += 1

                evaluations += 1
                if evaluations >= self.budget:
                    break

            self.adapt_and_refine(
                successful_trials, self.population_size, best_idx, population, fitness, func
            )
            population = new_population
            best_idx = np.argmin(fitness)

        return fitness[best_idx], population[best_idx]
