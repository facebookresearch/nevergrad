import numpy as np


class CGES:
    def __init__(self, budget, population_size=100, beta=0.15, mutation_strength=0.1, elitism=3):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.beta = beta  # Gradient influence in update
        self.mutation_strength = mutation_strength
        self.elitism = elitism  # Number of elites

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main evolutionary cycle
        while num_evals < self.budget:
            # Sort individuals based on fitness
            indices = np.argsort(fitness)
            elites = population[indices[: self.elitism]]

            new_population = np.zeros_like(population)
            new_population[: self.elitism] = elites  # Preserve elites directly

            # Generate new candidates
            for i in range(self.elitism, self.population_size):
                # Select random elite as a base for new candidate
                base_idx = np.random.choice(np.arange(self.elitism))
                base = population[indices[base_idx]]

                # Gradient direction towards best individual
                direction = best_individual - base

                # Mutation: normal perturbation
                mutation = np.random.normal(0, self.mutation_strength, self.dimension)

                # Create new individual
                new_individual = base + self.beta * direction + mutation
                new_individual = np.clip(new_individual, self.lb, self.ub)  # Ensure bounds are respected

                new_population[i] = new_individual

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            num_evals += self.population_size - self.elitism

            # Update the best individual found so far
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

        return best_fitness, best_individual
