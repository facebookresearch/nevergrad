import numpy as np


class HybridAdaptiveSearchStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set to 5
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        population_size = 150
        elite_size = int(0.05 * population_size)
        mutation_rate = 0.3
        mutation_scale = lambda t: 0.3 * np.exp(-0.001 * t)  # More gentle decaying mutation scale
        crossover_rate = 0.9
        local_search_prob = 0.1  # Probability of performing local search on new individuals

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
                    mutation = np.random.normal(0, mutation_scale(evaluations), self.dim)
                    child = np.clip(child + mutation, self.lb, self.ub)

                # Local search with a small probability
                if np.random.random() < local_search_prob:
                    direction = np.random.randn(self.dim)
                    step_size = 0.1
                    local_point = child + step_size * direction
                    local_point = np.clip(local_point, self.lb, self.ub)
                    if func(local_point) < func(child):
                        child = local_point

                new_population.append(child)

            new_population = np.vstack((new_population))
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
