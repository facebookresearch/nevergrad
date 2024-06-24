import numpy as np


class EnhancedPrecisionGuidedQuantumStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 350  # Further increased population size for extensive exploration
        self.elite_size = 70  # Increased elite size to mitigate premature convergence
        self.crossover_probability = 0.85  # Adjusted for a more balanced exploration-exploitation
        self.mutation_scale = 0.008  # Refined mutation scale for even finer adjustments
        self.quantum_mutation_scale = 0.03  # Reduced scale for smaller, precise quantum leaps
        self.quantum_probability = 0.25  # Increased probability for quantum mutations
        self.precision_boost_factor = 0.03  # Reduced boost factor for a smoother precision increase
        self.reactivity_factor = 0.015  # Reduced for more stable evolution
        self.recombination_rate = 0.25  # Increased rate for recombining elite solutions for diversity

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

        # Implement quantum-inspired recombination
        for i in range(num_offspring):
            if np.random.rand() < self.crossover_probability:
                p1, p2 = np.random.choice(elite.shape[0], 2, replace=False)
                offspring[i] = self.quantum_recombination(elite[p1], elite[p2])
            else:
                offspring[i] = elite[np.random.choice(elite.shape[0])]

            # Mutation controlled by remaining budget
            scale = self.mutation_scale + self.precision_boost_factor * np.log(remaining_budget + 1)
            offspring[i] += np.random.normal(0, scale, self.dim)

            # Quantum mutation with optimization-oriented control
            if np.random.rand() < self.quantum_probability:
                offspring[i] += np.random.normal(0, self.quantum_mutation_scale, self.dim)

            offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)

        return np.vstack([elite, offspring])

    def quantum_recombination(self, parent1, parent2):
        # Implement a quantum-inspired recombination mechanism
        mask = np.random.rand(self.dim) > 0.5
        child = np.where(mask, parent1, parent2)
        return child

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
