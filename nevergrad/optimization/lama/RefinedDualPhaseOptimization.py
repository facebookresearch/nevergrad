import numpy as np


class RefinedDualPhaseOptimization:
    def __init__(self, budget, dim=5, initial_exploration_ratio=0.6):
        self.budget = budget
        self.dim = dim
        self.bounds = np.array([-5.0, 5.0])
        self.initial_exploration_budget = int(budget * initial_exploration_ratio)
        self.exploitation_phase_budget = budget - self.initial_exploration_budget

    def initialize_population(self, size):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def mutate(self, individual, mutation_strength):
        mutant = individual + mutation_strength * np.random.randn(self.dim)
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        # Exploration Phase with Adaptive Mutation Strength
        population_size = 30
        mutation_strength = 0.8
        population = self.initialize_population(population_size)
        f_values = self.evaluate_population(func, population)
        evaluations = population_size
        best_score = np.min(f_values)
        best_individual = population[np.argmin(f_values)]

        while evaluations < self.initial_exploration_budget:
            new_population = []
            for individual in population:
                new_individual = self.mutate(individual, mutation_strength)
                new_population.append(new_individual)
            new_f_values = self.evaluate_population(func, new_population)
            evaluations += population_size

            combined_f_values = np.concatenate((f_values, new_f_values))
            combined_population = np.vstack((population, new_population))

            best_indices = np.argsort(combined_f_values)[:population_size]
            population = combined_population[best_indices]
            f_values = combined_f_values[best_indices]
            mutation_strength *= 0.95  # Reduce mutation strength gradually

        # Exploitation Phase with Local Search
        for _ in range(self.exploitation_phase_budget):
            perturbations = np.random.randn(self.dim) * 0.1
            candidate = np.clip(best_individual + perturbations, self.bounds[0], self.bounds[1])
            candidate_score = func(candidate)
            evaluations += 1

            if candidate_score < best_score:
                best_score = candidate_score
                best_individual = candidate

        return best_score, best_individual
