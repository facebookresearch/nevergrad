import numpy as np


class EnhancedHybridAdaptiveSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is explicitly set
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        population_size = 200
        elite_size = int(0.1 * population_size)
        mutation_rate = 0.2
        mutation_scale = lambda t: 0.1 * np.exp(-0.0005 * t)  # Slower decay to provide exploration longer
        crossover_rate = 0.95

        # Adapt local search probability based on remaining budget
        local_search_base_prob = 0.05
        local_search_decay_rate = 0.0001

        # Initialize population uniformly within the bounds
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        # Track best solution found
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Optimization loop
        while evaluations < self.budget:
            new_population = []
            elites_indices = np.argsort(fitness)[:elite_size]
            elites = population[elites_indices]

            # Generation loop
            while len(new_population) < population_size - elite_size:
                idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[idx1], population[idx2]

                # Crossover
                if np.random.random() < crossover_rate:
                    cross_point = np.random.randint(1, self.dim)
                    child = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                else:
                    child = parent1.copy()

                # Mutation
                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, mutation_scale(evaluations), self.dim)
                    child = np.clip(child + mutation, self.lb, self.ub)

                # Local search with adaptive probability
                local_search_prob = local_search_base_prob * np.exp(-local_search_decay_rate * evaluations)
                if np.random.random() < local_search_prob:
                    direction = np.random.randn(self.dim)  # Random direction
                    step_size = 0.05  # Smaller step size for finer local search
                    local_candidate = child + step_size * direction
                    local_candidate = np.clip(local_candidate, self.lb, self.ub)
                    if func(local_candidate) < func(child):
                        child = local_candidate

                new_population.append(child)

            new_population = np.vstack(new_population)
            new_fitness = np.array([func(x) for x in new_population])
            evaluations += len(new_population)

            # Combine elites with new generation
            population = np.vstack((elites, new_population))
            fitness = np.concatenate([fitness[elites_indices], new_fitness])

            # Update best solution if found
            current_best_idx = np.argmin(fitness)
            current_best_f = fitness[current_best_idx]
            if current_best_f < self.f_opt:
                self.f_opt = current_best_f
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
