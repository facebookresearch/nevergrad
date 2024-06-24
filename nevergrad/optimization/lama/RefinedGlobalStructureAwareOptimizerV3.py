import numpy as np


class RefinedGlobalStructureAwareOptimizerV3:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 150
        elite_size = 45
        evaluations = 0
        mutation_scale = 0.08
        adaptive_factor = 0.95
        recombination_prob = 0.95

        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(population_size):
                if np.random.rand() < recombination_prob:
                    indices = np.random.choice(population_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    child = x0 + mutation_scale * (x1 - x2)
                    child = np.clip(child, self.lb, self.ub)
                else:
                    child = population[i] + np.random.normal(0, mutation_scale, self.dim)
                    child = np.clip(child, self.lb, self.ub)

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < fitness[i]:
                    new_population.append(child)
                    new_fitness.append(child_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

            mutation_scale *= adaptive_factor
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]

            for idx in range(population_size - elite_size):
                if np.random.rand() < 0.4:  # Enhanced mutation within elites
                    replacement_idx = np.random.choice(elite_size)
                    population[idx] = elite_individuals[replacement_idx] + np.random.normal(
                        0, mutation_scale, self.dim
                    )
                    population[idx] = np.clip(population[idx], self.lb, self.ub)
                    fitness[idx] = func(population[idx])
                    evaluations += 1

            if evaluations % 250 == 0:  # More frequent and varied global structure mutations
                structure_scale = 0.3
                structure_population = np.random.normal(0, structure_scale, (population_size // 2, self.dim))
                structure_population = np.clip(
                    structure_population
                    + population[np.random.choice(population_size, population_size // 2)],
                    self.lb,
                    self.ub,
                )
                structure_fitness = np.array([func(ind) for ind in structure_population])
                evaluations += population_size // 2

                combined_population = np.concatenate((population, structure_population), axis=0)
                combined_fitness = np.concatenate((fitness, structure_fitness), axis=0)

                indices = np.argsort(combined_fitness)[:population_size]
                population = combined_population[indices]
                fitness = combined_fitness[indices]

        return self.f_opt, self.x_opt
