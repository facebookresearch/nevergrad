import numpy as np


class RefinedAdaptivePrecisionBalanceStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        population_size = 100
        elite_size = int(0.2 * population_size)
        mutation_rate = 0.6
        mutation_scale = 0.1
        crossover_rate = 0.7

        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Main optimization loop
        while evaluations < self.budget:
            new_population = []

            # Select elites to carry over to next generation
            elites_indices = np.argsort(fitness)[:elite_size]
            elites = population[elites_indices]

            # Generate the rest of the new population
            while len(new_population) < population_size - elite_size:
                idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[idx1], population[idx2]

                if np.random.random() < crossover_rate:
                    cross_point = np.random.randint(1, self.dim)
                    child = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                else:
                    child = parent1.copy()

                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, mutation_scale, self.dim)
                    child = np.clip(child + mutation, self.lb, self.ub)

                new_population.append(child)

            new_population = np.array(new_population)
            new_fitness = np.array([func(x) for x in new_population])
            evaluations += len(new_population)

            # Combine new population with elites
            population = np.vstack((elites, new_population))
            fitness = np.concatenate([fitness[elites_indices], new_fitness])

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            current_best_f = fitness[current_best_idx]
            if current_best_f < self.f_opt:
                self.f_opt = current_best_f
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt


# Example usage:
# optimizer = RefinedAdaptivePrecisionBalanceStrategy(budget=10000)
# best_value, best_solution = optimizer(func)
