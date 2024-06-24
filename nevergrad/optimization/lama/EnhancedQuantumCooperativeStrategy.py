import numpy as np


class EnhancedQuantumCooperativeStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.elite_size = 15  # Increased elite size for better exploitation
        self.crossover_fraction = 0.85  # Increased crossover fraction
        self.mutation_scale = 0.08  # Slightly reduced for more fine-grained search
        self.quantum_mutation_scale = 0.25  # Reduced quantum mutation scale
        self.quantum_probability = 0.03  # Reduced quantum probability for more stability

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_elite(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def crossover_and_mutate(self, parents, num_offspring):
        offspring = np.empty((num_offspring, self.dim))
        for i in range(num_offspring):
            if np.random.rand() < self.crossover_fraction:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                cross_point = np.random.randint(1, self.dim)
                offspring[i][:cross_point] = parents[p1][:cross_point]
                offspring[i][cross_point:] = parents[p2][cross_point:]
            else:
                offspring[i] = parents[np.random.randint(len(parents))]

            # Mutation strategy
            if np.random.rand() < self.quantum_probability:
                mutation_shift = np.random.normal(0, self.quantum_mutation_scale, self.dim)
            else:
                mutation_shift = np.random.normal(0, self.mutation_scale, self.dim)
            offspring[i] += mutation_shift
            offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)
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

            elite_population, elite_fitness = self.select_elite(population, fitness)
            num_offspring = self.population_size - self.elite_size
            offspring = self.crossover_and_mutate(elite_population, num_offspring)

            population = np.vstack((elite_population, offspring))

        return best_score, best_solution
