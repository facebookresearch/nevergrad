import numpy as np


class ADEAS:
    def __init__(self, budget):
        self.budget = budget
        self.initial_population_size = 20
        self.dimension = 5
        self.low = -5.0
        self.high = 5.0
        self.T_initial = 1.0
        self.T_min = 0.01
        self.decay_rate = 0.95

    def initialize(self):
        population_size = self.initial_population_size
        population = np.random.uniform(self.low, self.high, (population_size, self.dimension))
        F = np.random.normal(0.5, 0.1, population_size)
        CR = np.random.normal(0.9, 0.05, population_size)
        return population, F, CR, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, population, F):
        mutant = np.zeros_like(population)
        for i in range(len(population)):
            indices = np.random.choice(len(population), 3, replace=False)
            x1, x2, x3 = population[indices]
            mutant_vector = x1 + F[i] * (x2 - x3)
            mutant[i] = np.clip(mutant_vector, self.low, self.high)
        return mutant

    def crossover(self, population, mutant, CR):
        crossover = np.where(
            np.random.rand(len(population), self.dimension) < CR[:, None], mutant, population
        )
        return crossover

    def select(self, population, fitness, trial_population, trial_fitness, F, CR):
        improved = trial_fitness < fitness
        population[improved] = trial_population[improved]
        fitness[improved] = trial_fitness[improved]

        # Update F and CR adaptively
        F[improved] = np.clip(F[improved] * 1.1, 0.1, 1.0)
        CR[improved] = np.clip(CR[improved] * 0.95, 0.1, 1.0)
        F[~improved] = np.clip(F[~improved] * 0.9, 0.1, 1.0)
        CR[~improved] = np.clip(CR[~improved] * 1.05, 0.1, 1.0)

        return population, fitness, F, CR

    def adaptive_local_search(self, individual, func, T):
        for _ in range(10):
            neighbor = individual + np.random.normal(0, T, self.dimension)
            neighbor = np.clip(neighbor, self.low, self.high)
            if func(neighbor) < func(individual):
                individual = neighbor
        return individual

    def __call__(self, func):
        population, F, CR, population_size = self.initialize()
        fitness = self.evaluate(population, func)
        T = self.T_initial
        evaluations = population_size

        while evaluations < self.budget:
            mutant = self.mutation(population, F)
            trial_population = self.crossover(population, mutant, CR)
            trial_fitness = self.evaluate(trial_population, func)
            evaluations += len(trial_population)

            population, fitness, F, CR = self.select(
                population, fitness, trial_population, trial_fitness, F, CR
            )

            # Dynamic population adjustment
            if np.std(fitness) < np.mean(fitness) * 0.1 and len(population) < 40:
                additional_members = np.random.uniform(self.low, self.high, (10, self.dimension))
                population = np.vstack([population, additional_members])
                additional_fitness = self.evaluate(additional_members, func)
                fitness = np.concatenate([fitness, additional_fitness])
                evaluations += len(additional_members)
                F = np.concatenate([F, np.random.normal(0.5, 0.1, 10)])
                CR = np.concatenate([CR, np.random.normal(0.9, 0.05, 10)])

            # Local search with adaptive temperature
            selected_indices = np.random.choice(len(population), size=5, replace=False)
            for idx in selected_indices:
                population[idx] = self.adaptive_local_search(population[idx], func, T)
                fitness[idx] = func(population[idx])
            T = max(T * self.decay_rate, self.T_min)

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        return best_fitness, best_individual
