import numpy as np


class RefinedQuantumInformedGradientOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set to 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 150  # A smaller population for more focused search
        mutation_factor = 0.5  # Reduced mutation for more controlled exploration
        crossover_prob = 0.5  # Reduced crossover probability for more stable descent
        learning_rate = 0.01  # Initial learning rate for gradient-based steps

        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Main optimization loop
        while current_budget < self.budget:
            new_population = np.empty_like(population)
            gradients = np.zeros_like(population)

            # Estimating gradients using central difference method
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                base_ind = population[i]
                for d in range(self.dim):
                    perturbed_ind_plus = np.array(base_ind)
                    perturbed_ind_minus = np.array(base_ind)
                    perturbed_ind_plus[d] += learning_rate
                    perturbed_ind_minus[d] -= learning_rate

                    if current_budget + 2 <= self.budget:
                        fitness_plus = func(perturbed_ind_plus)
                        fitness_minus = func(perturbed_ind_minus)
                        current_budget += 2
                        gradient = (fitness_plus - fitness_minus) / (2 * learning_rate)
                        gradients[i, d] = gradient

            # Apply learned gradients, mutation, and crossover to form new population
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                child = population[i] - learning_rate * gradients[i]  # Applying gradient
                child += np.random.randn(self.dim) * mutation_factor  # Applying mutation

                # Crossover operation
                if np.random.rand() < crossover_prob:
                    partner_idx = np.random.randint(population_size)
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    child = child * crossover_mask + population[partner_idx] * (1 - crossover_mask)

                child = np.clip(child, self.lower_bound, self.upper_bound)
                child_fitness = func(child)
                current_budget += 1

                if child_fitness < fitness[i]:
                    new_population[i] = child
                    fitness[i] = child_fitness
                else:
                    new_population[i] = population[i]

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

            population = new_population

            # Adaptive adjustments to learning rate and mutation factor
            mutation_factor *= 0.95  # Gradual decay
            learning_rate *= 0.99  # Decreasing learning rate to refine convergence

        return best_fitness, best_solution
