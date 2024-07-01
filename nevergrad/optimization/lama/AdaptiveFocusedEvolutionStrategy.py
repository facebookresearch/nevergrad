import numpy as np


class AdaptiveFocusedEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initial populations and parameters
        population_size = 100
        sigma = 0.5  # Initial standard deviation for Gaussian mutation
        elite_size = max(1, int(population_size * 0.05))
        learning_rate = 0.1  # Learning rate for adaptive sigma

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        for _ in range(int(self.budget / population_size)):
            # Elitism: keep the best solutions
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population = population[elite_indices].copy()
            new_fitness = fitness[elite_indices].copy()

            # Generate new population based on best solutions
            for i in range(elite_size, population_size):
                # Select parent from elite randomly
                parent_index = np.random.choice(elite_indices)
                parent = population[parent_index]

                # Apply adaptive Gaussian mutation
                offspring = parent + np.random.normal(0, sigma, self.dim)
                offspring = np.clip(offspring, self.lower_bound, self.upper_bound)
                offspring_fitness = func(offspring)

                # Replace if better
                if offspring_fitness < fitness[parent_index]:
                    new_population = np.vstack([new_population, offspring])
                    new_fitness = np.append(new_fitness, offspring_fitness)
                else:
                    new_population = np.vstack([new_population, parent])
                    new_fitness = np.append(new_fitness, fitness[parent_index])

            # Update population
            population = new_population
            fitness = new_fitness

            # Adaptive mutation step size
            sigma *= np.exp(learning_rate * (1.0 - np.mean(fitness) / best_fitness))

            # Update the best solution found
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_solution = population[current_best_index]

        return best_fitness, best_solution
