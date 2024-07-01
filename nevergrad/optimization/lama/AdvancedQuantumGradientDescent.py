import numpy as np


class AdvancedQuantumGradientDescent:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.elite_size = 10
        self.mutation_scale = 0.1
        self.quantum_probability = 0.15
        self.gradient_steps = 5

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_elite(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def mutate(self, individual):
        if np.random.rand() < self.quantum_probability:
            # Quantum mutation
            mutation = np.random.normal(0, self.mutation_scale, self.dim)
        else:
            # Standard mutation
            mutation = np.random.normal(0, self.mutation_scale / 10, self.dim)
        new_individual = np.clip(individual + mutation, self.lower_bound, self.upper_bound)
        return new_individual

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(func, population)
        best_idx = np.argmin(fitness)
        best_score = fitness[best_idx]
        best_solution = population[best_idx].copy()

        evaluations = self.population_size

        while evaluations < self.budget:
            elite_population, elite_fitness = self.select_elite(population, fitness)
            new_population = np.zeros_like(population)

            # Crossover and mutation:
            for i in range(self.population_size):
                if i < self.elite_size:
                    new_population[i] = elite_population[i]
                else:
                    parent = elite_population[np.random.randint(self.elite_size)]
                    new_population[i] = self.mutate(parent)

            # Evaluate new population
            new_fitness = self.evaluate(func, new_population)

            # Update best solution
            min_idx = np.argmin(new_fitness)
            if new_fitness[min_idx] < best_score:
                best_score = new_fitness[min_idx]
                best_solution = new_population[min_idx].copy()

            population = new_population
            fitness = new_fitness
            evaluations += self.population_size

        return best_score, best_solution
