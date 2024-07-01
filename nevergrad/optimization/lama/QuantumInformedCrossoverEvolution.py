import numpy as np


class QuantumInformedCrossoverEvolution:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.elite_size = 20
        self.offspring_size = 80
        self.mutation_scale = 0.01
        self.crossover_prob = 0.95
        self.mutation_prob = 0.1
        self.quantum_probability = 0.15  # Probability to perform quantum-inspired jump

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_survivors(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_prob:
            cross_point = np.random.randint(1, self.dim)
            child = np.empty(self.dim)
            child[:cross_point] = parent1[:cross_point]
            child[cross_point:] = parent2[cross_point:]
            return child
        return parent1 if np.random.rand() < 0.5 else parent2

    def mutate(self, individual):
        if np.random.rand() < self.mutation_prob:
            mutation_points = np.random.randint(0, self.dim)
            individual[mutation_points] += np.random.normal(0, self.mutation_scale)
            individual = np.clip(individual, self.lower_bound, self.upper_bound)
        return individual

    def quantum_jump(self, individual):
        if np.random.rand() < self.quantum_probability:
            return np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        return individual

    def reproduce(self, parents):
        offspring = np.empty((self.offspring_size, self.dim))
        num_parents = len(parents)
        for i in range(self.offspring_size):
            p1, p2 = np.random.choice(num_parents, 2, replace=False)
            child = self.crossover(parents[p1], parents[p2])
            child = self.mutate(child)
            child = self.quantum_jump(child)
            offspring[i] = child
        return offspring

    def __call__(self, func):
        population = self.initialize_population()
        best_score = float("inf")
        best_solution = None
        evaluations_consumed = 0

        while evaluations_consumed < self.budget:
            fitness = self.evaluate(func, population)
            evaluations_consumed += len(population)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_score:
                best_score = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            if evaluations_consumed >= self.budget:
                break

            elite_population, elite_fitness = self.select_survivors(population, fitness)

            offspring = self.reproduce(elite_population)

            population = np.vstack((elite_population, offspring))

        return best_score, best_solution
