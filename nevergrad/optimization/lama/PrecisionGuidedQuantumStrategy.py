import numpy as np


class PrecisionGuidedQuantumStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 300  # Increased population size for broader search
        self.elite_size = 60  # Larger elite size to retain more high-quality solutions
        self.crossover_probability = 0.9  # Slightly increased for better diversity
        self.mutation_scale = 0.01  # More precise mutation scale for finer adjustments
        self.quantum_mutation_scale = 0.05  # Lower scale for precise quantum leaps
        self.quantum_probability = 0.2  # Higher frequency for quantum mutations
        self.precision_boost_factor = 0.05  # Boost factor for precision in later stages
        self.reactivity_factor = 0.02  # Lower reactivity factor for stable evolution
        self.recombination_rate = 0.2  # Rate for recombining elite solutions

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_elite(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def evolve_population(self, elite, remaining_budget):
        num_offspring = self.population_size - self.elite_size
        offspring = np.empty((num_offspring, self.dim))

        for i in range(num_offspring):
            if np.random.rand() < self.crossover_probability:
                p1, p2 = np.random.choice(elite.shape[0], 2, replace=False)
                cross_point = np.random.randint(1, self.dim)
                offspring[i][:cross_point] = elite[p1][:cross_point]
                offspring[i][cross_point:] = elite[p2][cross_point:]
            else:
                offspring[i] = elite[np.random.choice(elite.shape[0])]

            # Apply deterministic mutation for precision
            scale = self.mutation_scale + self.precision_boost_factor * np.log(remaining_budget + 1)
            offspring[i] += np.random.normal(0, scale, self.dim)

            # Apply quantum mutation with controlled probability
            if np.random.rand() < self.quantum_probability:
                offspring[i] += np.random.normal(0, self.quantum_mutation_scale, self.dim)

            offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)

        return np.vstack([elite, offspring])

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
            remaining_budget = self.budget - evaluations_consumed
            population = self.evolve_population(elite_population, remaining_budget)

        return best_score, best_solution
