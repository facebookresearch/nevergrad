import numpy as np


class OptimalQuantumSynergyStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 400  # Enhanced population size for wider exploration
        self.elite_size = 80  # Expanded elite pool to preserve more high-quality solutions
        self.crossover_probability = 0.9  # Increased probability to promote genetic diversity
        self.mutation_scale = 0.005  # Further refined mutation for micro adjustments
        self.quantum_mutation_scale = 0.02  # Optimal scale for effective quantum leaps
        self.quantum_probability = 0.3  # Higher chance for quantum mutations to foster innovative solutions
        self.precision_boost_factor = 0.02  # Optimally tuned boost factor for precision enhancement
        self.reactivity_factor = 0.01  # Minimized to stabilize evolution dynamics
        self.recombination_rate = 0.3  # Enhanced for more frequent recombination among elites

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
                offspring[i] = self.optimal_quantum_recombination(elite[p1], elite[p2])
            else:
                offspring[i] = elite[np.random.choice(elite.shape[0])]

            scale = self.mutation_scale + self.precision_boost_factor * np.log(remaining_budget + 1)
            offspring[i] += np.random.normal(0, scale, self.dim)

            if np.random.rand() < self.quantum_probability:
                offspring[i] += np.random.normal(0, self.quantum_mutation_scale, self.dim)

            offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)

        return np.vstack([elite, offspring])

    def optimal_quantum_recombination(self, parent1, parent2):
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
