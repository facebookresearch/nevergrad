import numpy as np


class QuantumInformedGradientOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5.
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 200  # Adjusted population size for effective exploration and exploitation
        mutation_factor = 0.8  # Mutation factor for controlled exploration
        crossover_prob = 0.7  # Crossover probability to maintain diversity
        learning_rate = 0.01  # Learning rate for gradient descent steps

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while current_budget < self.budget:
            new_population = np.empty_like(population)
            gradients = np.zeros_like(population)

            # Perform a quasi-gradient estimation using finite differences
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                base_ind = population[i]
                for d in range(self.dim):
                    perturbed_ind = np.array(base_ind)
                    perturbed_ind[d] += learning_rate
                    if current_budget < self.budget:
                        perturbed_fitness = func(perturbed_ind)
                        current_budget += 1
                        gradient = (perturbed_fitness - fitness[i]) / learning_rate
                        gradients[i, d] = gradient

            # Update steps based on gradients and mutation
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Apply mutation and crossover
                if np.random.rand() < crossover_prob:
                    partner_idx = np.random.randint(population_size)
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    child = population[i] * crossover_mask + population[partner_idx] * (1 - crossover_mask)
                else:
                    child = population[i]

                mutation = np.random.randn(self.dim) * mutation_factor
                child -= learning_rate * gradients[i] + mutation
                child = np.clip(child, self.lower_bound, self.upper_bound)

                child_fitness = func(child)
                current_budget += 1

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adaptive updates to parameters
            mutation_factor *= 0.99  # Slowly decrease mutation factor
            crossover_prob = max(0.5, crossover_prob - 0.01)  # Gradually decrease crossover probability

        return best_fitness, best_solution
