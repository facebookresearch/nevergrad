import numpy as np


class EnhancedQuantumInspiredHybridOptimizer:
    def __init__(self, budget, dim=5, population_size=50, elite_size=10):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.elite_size = elite_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.tournament_size = 5
        self.mutation_prob = 0.2
        self.learning_rate = 0.1
        self.alpha = 0.75  # Factor to adjust mutation probability dynamically

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(x) for x in population])

    def tournament_selection(self, population, fitnesses):
        selected_indices = np.random.randint(
            0, self.population_size, (self.population_size, self.tournament_size)
        )
        selected_fitnesses = fitnesses[selected_indices]
        winners_indices = selected_indices[
            np.arange(self.population_size), np.argmin(selected_fitnesses, axis=1)
        ]
        return population[winners_indices]

    def mutate(self, population):
        mutation_mask = np.random.rand(self.population_size, self.dim) < self.mutation_prob * self.alpha
        gaussian_perturbations = np.random.normal(0, self.learning_rate, (self.population_size, self.dim))
        mutated_population = population + mutation_mask * gaussian_perturbations
        return np.clip(mutated_population, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(func, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        iterations = self.population_size
        while iterations < self.budget:
            selected = self.tournament_selection(population, fitness)
            mutated = self.mutate(selected)
            mutated_fitness = self.evaluate_population(func, mutated)

            combined_population = np.vstack((population, mutated))
            combined_fitness = np.concatenate((fitness, mutated_fitness))

            top_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[top_indices]
            fitness = combined_fitness[top_indices]

            if np.min(fitness) < best_fitness:
                best_idx = np.argmin(fitness)
                best_individual = population[best_idx]
                best_fitness = fitness[best_idx]

            iterations += self.population_size
            # Update mutation probability dynamically to balance exploration and exploitation
            self.mutation_prob *= 1 - self.alpha

        return best_fitness, best_individual
