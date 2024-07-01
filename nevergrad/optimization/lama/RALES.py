import numpy as np


class RALES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 100  # Increased population size for better exploration
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        num_evals = population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Evolution parameters
        learning_rate = 0.05  # Reduced learning rate for more stable convergence
        global_learning_rate = 0.25  # Increased global learning rate for stronger elite attraction
        mutation_strength = 0.7  # Increased initial mutation strength for wider initial search
        mutation_decay = 0.95  # Slower mutation decay to maintain exploration longer
        elite_fraction = 0.2  # Increased elite fraction for a stronger focus on best solutions
        elite_size = int(population_size * elite_fraction)

        while num_evals < self.budget:
            elite_indices = np.argsort(fitness)[:elite_size]
            global_mean = np.mean(population[elite_indices], axis=0)

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                step = mutation_strength * np.random.randn(self.dimension)
                if i in elite_indices:
                    # Elites undergo less mutation
                    step *= 0.5

                individual = population[i] + step
                individual = np.clip(individual, self.lower_bound, self.upper_bound)

                # Global pull move towards the mean of elites
                individual = individual + global_learning_rate * (global_mean - individual)
                individual_fitness = func(individual)
                num_evals += 1

                # Selection process
                if individual_fitness < fitness[i]:
                    population[i] = individual
                    fitness[i] = individual_fitness
                    if individual_fitness < best_fitness:
                        best_fitness = individual_fitness
                        best_individual = individual.copy()

            # Decay the mutation strength
            mutation_strength *= mutation_decay

        return best_fitness, best_individual
