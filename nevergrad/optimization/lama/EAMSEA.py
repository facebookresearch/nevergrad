import numpy as np


class EAMSEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.last_best_fitness = np.inf
        self.stagnation_counter = 0
        self.adaptation_rate = 0.05

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index, learning_rate):
        F = 0.5 * learning_rate
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = np.random.choice(np.delete(np.arange(self.population_size), best_index), 3, replace=False)
            a, b, c = population[idxs]
            mutant_vector = a + F * (b - c)
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def adaptive_crossover(self, target, mutant):
        CR = 0.1 + 0.5 * np.random.rand()
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def calculate_learning_rate(self, current_best_fitness, population_var):
        if current_best_fitness < self.last_best_fitness:
            rate_increase = self.adaptation_rate / (1 + population_var)
        else:
            rate_increase = -self.adaptation_rate * population_var
        return max(0.1, min(0.9, rate_increase))

    def multi_neighborhood_search(self, best_individual, func):
        scales = np.linspace(0.1, 1.0, 10)
        candidates = [best_individual + scale * np.random.normal(0, 1, self.dimension) for scale in scales]
        candidates = np.clip(candidates, self.bounds[0], self.bounds[1])
        fitnesses = self.evaluate(candidates, func)
        best_local_idx = np.argmin(fitnesses)
        return candidates[best_local_idx]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)

        while evaluations < self.budget:
            current_best_fitness = fitness[best_index]
            population_var = np.var(fitness)
            learning_rate = self.calculate_learning_rate(current_best_fitness, population_var)
            mutants = self.mutate(population, best_index, learning_rate)
            trials = np.array(
                [self.adaptive_crossover(population[i], mutants[i]) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += len(trials)

            for i in range(self.population_size):
                if fitness_trials[i] < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = fitness_trials[i]

            best_index = np.argmin(fitness)
            if fitness[best_index] > self.last_best_fitness:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
                self.last_best_fitness = fitness[best_index]

            if self.stagnation_counter > 20:
                population[best_index] = self.multi_neighborhood_search(population[best_index], func)

        return fitness[best_index], population[best_index]
