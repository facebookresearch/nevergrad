import numpy as np


class PrecisionTunedHybridSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        population_size = 250
        elite_size = int(0.15 * population_size)
        mutation_rate = 0.25
        mutation_scale = lambda t: 0.1 * np.exp(-0.0003 * t)
        crossover_rate = 0.90

        local_search_prob_base = 0.1
        local_search_decay = 0.0002

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            elites_indices = np.argsort(fitness)[:elite_size]
            elites = population[elites_indices]

            while len(new_population) < population_size - elite_size:
                idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                parent1, parent2 = population[idx1], population[idx2]

                if np.random.random() < crossover_rate:
                    point = np.random.randint(1, self.dim)
                    child = np.concatenate([parent1[:point], parent2[point:]])
                else:
                    child = parent1.copy()

                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, mutation_scale(evaluations), self.dim)
                    child = np.clip(child + mutation, self.lb, self.ub)

                local_search_prob = local_search_prob_base * np.exp(-local_search_decay * evaluations)
                if np.random.random() < local_search_prob:
                    direction = np.random.randn(self.dim)
                    step = 0.03
                    candidate = child + step * direction
                    candidate = np.clip(candidate, self.lb, self.ub)
                    if func(candidate) < func(child):
                        child = candidate

                new_population.append(child)

            new_population = np.vstack(new_population)
            new_fitness = np.array([func(x) for x in new_population])
            evaluations += len(new_population)

            population = np.vstack((elites, new_population))
            fitness = np.concatenate([fitness[elites_indices], new_fitness])

            current_best_idx = np.argmin(fitness)
            current_best_f = fitness[current_best_idx]
            if current_best_f < self.f_opt:
                self.f_opt = current_best_f
                self.x_opt = population[current_best_idx]

        return self.f_opt, self.x_opt
