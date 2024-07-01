import numpy as np


class EnhancedQuantumSynergyStrategyV2:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 500  # Increased population size for broader exploration
        self.elite_size = 100  # Larger elite pool to maintain more successful individuals
        self.crossover_probability = 0.95  # Slightly increased crossover probability
        self.mutation_scale = 0.002  # Decreased mutation scale for finer adaptations
        self.quantum_mutation_scale = 0.015  # Reduced quantum mutation scale for controlled exploration
        self.quantum_probability = 0.35  # Increased probability to enhance quantum mutation effects
        self.precision_boost_factor = 0.015  # Reduced boost factor for better precision in later stages
        self.reactivity_factor = 0.008  # Further reduced to stabilize the dynamic changes
        self.recombination_rate = 0.35  # Enhanced recombination rate among elites

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
                offspring[i] = self.optimal_quantum_recombination(elite[p1], elite[p2], remaining_budget)
            else:
                offspring[i] = elite[np.random.choice(elite.shape[0])]

            scale = self.mutation_scale + self.precision_boost_factor * np.log(remaining_budget + 1)
            offspring[i] += np.random.normal(0, scale, self.dim)

            if np.random.rand() < self.quantum_probability:
                offspring[i] += np.random.normal(0, self.quantum_mutation_scale, self.dim)

            offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)

        return np.vstack([elite, offspring])

    def optimal_quantum_recombination(self, parent1, parent2, remaining_budget):
        blend_factor = self.reactivity_factor * np.exp(-remaining_budget / self.budget)
        child = blend_factor * parent1 + (1 - blend_factor) * parent2
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
