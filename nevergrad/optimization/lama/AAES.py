import numpy as np


class AAES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.F = 0.8  # Mutation factor
        self.CR = 0.9  # Crossover rate
        self.stagnation_threshold = 20  # Threshold for enhanced local search and rejuvenation
        self.no_improvement_intervals = 0
        self.momentum_F = 0.95
        self.momentum_CR = 0.05

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, idx, population):
        indices = np.delete(np.arange(self.population_size), idx)
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.CR
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

    def update_parameters(self, successes, trials):
        self.F = max(0.1, self.F * (self.momentum_F if successes / trials < 0.2 else 1.05))
        self.CR = 0.1 + 0.8 * (self.momentum_CR * successes + (1 - self.momentum_CR) * trials)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_score = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.copy(population)
            successes = 0

            for i in range(self.population_size):
                mutant = self.mutation(i, population)
                trial = self.crossover(population[i], mutant)
                f_trial = func(trial)
                f_target = fitness[i]

                if f_trial < f_target:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    successes += 1

                evaluations += 1
                if evaluations >= self.budget:
                    break

            if successes == 0:
                self.no_improvement_intervals += 1
            if self.no_improvement_intervals >= self.stagnation_threshold:
                population[np.argsort(fitness)[-10:]] = self.initialize_population()[
                    :10
                ]  # Rejuvenate worst 10
                fitness[np.argsort(fitness)[-10:]] = self.evaluate(
                    population[np.argsort(fitness)[-10:]], func
                )
                self.no_improvement_intervals = 0

            self.update_parameters(successes, self.population_size)
            population = new_population
            current_best = np.argmin(fitness)
            if fitness[current_best] < best_score:
                best_idx = current_best
                best_score = fitness[best_idx]

        return fitness[best_idx], population[best_idx]
