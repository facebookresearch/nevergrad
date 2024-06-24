import numpy as np


class SGE:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 100
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Introducing learning rate and momentum for evolution strategy
        learning_rate = 0.01
        momentum = 0.9
        velocity = np.zeros_like(population)

        while num_evals < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Decay learning rate and adapt momentum
            learning_rate *= 0.99
            current_momentum = momentum * (1 - 0.95 * (num_evals / self.budget))

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                # Gradient approximation via finite differences
                perturbation = np.random.normal(0, 1, self.dimension)
                candidate_plus = population[i] + learning_rate * perturbation
                candidate_minus = population[i] - learning_rate * perturbation

                candidate_plus = np.clip(candidate_plus, self.lower_bound, self.upper_bound)
                candidate_minus = np.clip(candidate_minus, self.lower_bound, self.upper_bound)

                fitness_plus = func(candidate_plus)
                fitness_minus = func(candidate_minus)
                num_evals += 2

                # Compute approximate gradient
                gradient = (fitness_minus - fitness_plus) / (2 * learning_rate * perturbation)
                velocity[i] = current_momentum * velocity[i] - learning_rate * gradient
                candidate = population[i] + velocity[i]
                candidate = np.clip(candidate, self.lower_bound, self.upper_bound)

                trial_fitness = func(candidate)
                num_evals += 1

                # Keep track of new population and fitness
                new_population[i] = candidate if trial_fitness < fitness[i] else population[i]
                new_fitness[i] = trial_fitness if trial_fitness < fitness[i] else fitness[i]

            # Update population and best individual
            population[:] = new_population
            fitness[:] = new_fitness
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness = fitness[best_idx]
                best_individual = population[best_idx].copy()

        return best_fitness, best_individual
