import numpy as np


class AdaptiveQuantumInformedGradientEnhancer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set to 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 100  # Adjusted population size for refined search
        mutation_factor = 0.8  # Initially higher mutation for broader search
        crossover_prob = 0.7  # Initially higher crossover probability for diverse search patterns
        learning_rate = 0.1  # Starting learning rate for gradient-based steps

        # Initialize population within bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Optimization loop
        while current_budget < self.budget:
            gradients = np.zeros_like(population)

            # Gradient estimation with central difference method
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

            new_population = population.copy()  # Start with a copy of the current population

            # Generate new solutions based on gradients, mutation, and crossover
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Apply gradient descent
                child = population[i] - learning_rate * gradients[i]

                # Mutation step
                child += np.random.randn(self.dim) * mutation_factor

                # Crossover step
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

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

            population = new_population

            # Adaptively adjust learning rate, mutation factor, and crossover probability
            mutation_factor *= 0.98  # Gradual decline in mutation factor
            learning_rate *= 0.98  # Gradual decrease in learning rate
            crossover_prob *= 0.98  # Reduce the crossover probability to stabilize final convergence

        return best_fitness, best_solution
