import numpy as np


class AdaptiveQuantumGradientEnhancedOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 40  # Increased population size for better exploration
        mutation_scale = 0.3  # Initial mutation scale
        crossover_probability = 0.8  # Higher initial crossover probability
        learning_rate = 0.01  # Learning rate for gradient descent

        # Initialize population within bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Optimization loop
        while current_budget < self.budget:
            new_population = np.copy(population)

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Mutation: Quantum-inspired normal perturbation
                mutation = np.random.normal(0, mutation_scale, self.dim)
                candidate = population[i] + mutation

                # Gradient calculation for local search
                grad = np.zeros(self.dim)
                for d in range(self.dim):
                    if current_budget + 2 > self.budget:
                        break
                    perturb = np.zeros(self.dim)
                    perturb[d] = learning_rate
                    f_plus = func(population[i] + perturb)
                    f_minus = func(population[i] - perturb)
                    current_budget += 2

                    grad[d] = (f_plus - f_minus) / (2 * learning_rate)

                # Update candidate using gradient information
                candidate -= learning_rate * grad

                # Crossover with random individual
                if np.random.rand() < crossover_probability:
                    partner_index = np.random.randint(population_size)
                    mask = np.random.rand(self.dim) < 0.5  # Uniform mask
                    candidate[mask] = population[partner_index][mask]

                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                candidate_fitness = func(candidate)
                current_budget += 1

                # Greedy selection
                if candidate_fitness < fitness[i]:
                    new_population[i] = candidate
                    fitness[i] = candidate_fitness

                # Update best found solution
                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate

            population = new_population
            # Adaptive update of mutation scale and crossover probability
            mutation_scale *= 0.98
            crossover_probability *= 0.97

        return best_fitness, best_solution
