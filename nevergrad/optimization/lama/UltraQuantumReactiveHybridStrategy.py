import numpy as np


class UltraQuantumReactiveHybridStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 250  # Further increased population size to boost exploration
        self.elite_size = 40  # Increased elite size to ensure retention of best solutions
        self.crossover_fraction = 0.85  # Adjusted to enhance genetic diversity
        self.mutation_scale = 0.02  # Further refined mutation scale for precise local searches
        self.quantum_mutation_scale = 0.1  # Reduced for controlled explorative steps
        self.quantum_probability = 0.15  # Increased for more frequent quantum mutations
        self.reactivity_factor = 0.03  # Further refined for stable mutation adaptation
        self.adaptive_quantum_boost = 0.03  # Increased boost factor for enhanced late-stage exploration
        self.hybridization_rate = 0.1  # New: Rate at which we hybridize solutions from elite and random

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_elite(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def crossover_and_mutate(self, parents, num_offspring, iteration):
        offspring = np.empty((num_offspring, self.dim))
        parent_indices = np.arange(len(parents))
        for i in range(num_offspring):
            if np.random.rand() < self.crossover_fraction:
                p1, p2 = np.random.choice(parent_indices, 2, replace=False)
                cross_point = np.random.randint(1, self.dim)
                offspring[i][:cross_point] = parents[p1][:cross_point]
                offspring[i][cross_point:] = parents[p2][cross_point:]
            else:
                offspring[i] = parents[np.random.choice(parent_indices)]

            dynamic_scale = self.mutation_scale / (1 + iteration * self.reactivity_factor)
            dynamic_quantum_scale = (
                self.quantum_mutation_scale + iteration * self.adaptive_quantum_boost
            ) / (1 + iteration * self.reactivity_factor)

            if np.random.rand() < self.quantum_probability:
                mutation_shift = np.random.normal(0, dynamic_quantum_scale, self.dim)
            else:
                mutation_shift = np.random.normal(0, dynamic_scale, self.dim)
            offspring[i] += mutation_shift
            offspring[i] = np.clip(offspring[i], self.lower_bound, self.upper_bound)
        return offspring

    def hybridize(self, elite, random_selection):
        hybrid_count = int(self.hybridization_rate * len(elite))
        hybrids = np.empty((hybrid_count, self.dim))
        for h in range(hybrid_count):
            elite_member = elite[np.random.randint(len(elite))]
            random_member = random_selection[np.random.randint(len(random_selection))]
            mix_ratio = np.random.rand()
            hybrids[h] = mix_ratio * elite_member + (1 - mix_ratio) * random_member
        return hybrids

    def __call__(self, func):
        population = self.initialize_population()
        best_score = float("inf")
        best_solution = None
        evaluations_consumed = 0

        iteration = 0
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
            num_offspring = self.population_size - self.elite_size - len(elite_population)
            offspring = self.crossover_and_mutate(elite_population, num_offspring, iteration)

            random_selection = self.initialize_population()[: len(elite_population)]
            hybrids = self.hybridize(elite_population, random_selection)

            population = np.vstack((elite_population, offspring, hybrids))
            iteration += 1

        return best_score, best_solution
