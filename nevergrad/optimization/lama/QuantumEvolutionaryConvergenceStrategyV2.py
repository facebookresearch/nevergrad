import numpy as np


class QuantumEvolutionaryConvergenceStrategyV2:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 200  # Refined population size for better focus
        self.elite_size = 40  # Optimal size for elite selection refined
        self.crossover_probability = 0.85  # Slightly more aggressive crossover
        self.mutation_scale = 0.005  # Fine-tuned mutation scale for more accurate local searches
        self.quantum_mutation_scale = 0.015  # Balanced quantum mutation scale for exploration
        self.quantum_probability = 0.2  # Probability for quantum mutation slightly adjusted
        self.recombination_rate = 0.6  # Increased rate for better population diversity

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_elite(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            alpha = np.random.rand()
            return alpha * parent1 + (1 - alpha) * parent2
        else:
            return parent1 if np.random.rand() < 0.5 else parent2

    def mutate(self, individual):
        mutation_mask = np.random.rand(self.dim) < self.mutation_scale
        individual[mutation_mask] += np.random.normal(0, self.mutation_scale, np.sum(mutation_mask))
        if np.random.rand() < self.quantum_probability:
            quantum_mutation_mask = np.random.rand(self.dim) < self.quantum_mutation_scale
            individual[quantum_mutation_mask] += np.random.normal(
                0, self.quantum_mutation_scale, np.sum(quantum_mutation_mask)
            )
        return np.clip(individual, self.lower_bound, self.upper_bound)

    def evolve_population(self, elite, remaining_budget):
        num_offspring = self.population_size - self.elite_size
        offspring = np.empty((num_offspring, self.dim))

        for i in range(num_offspring):
            p1, p2 = np.random.choice(elite.shape[0], 2, replace=False)
            child = self.crossover(elite[p1], elite[p2])
            child = self.mutate(child)
            offspring[i] = child

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
