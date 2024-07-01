import numpy as np


class EnhancedGlobalStructureAdaptiveEvolver:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 250
        elite_size = 80
        evaluations = 0
        mutation_scale = 0.1
        adaptive_factor = 0.95
        recombination_prob = 0.75
        innovators_factor = 0.1  # Proportion of population for extensive exploration

        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            # Recombination and mutation within elites
            new_population = elite_individuals.copy()
            new_fitness = elite_fitness.copy()
            for _ in range(population_size - elite_size):
                if np.random.rand() < recombination_prob:
                    indices = np.random.choice(elite_size, 3, replace=False)
                    x0, x1, x2 = elite_individuals[indices]
                    child = x0 + mutation_scale * (x1 - x2)
                    child = np.clip(child, self.lb, self.ub)
                else:
                    idx = np.random.choice(elite_size)
                    child = elite_individuals[idx] + np.random.normal(0, mutation_scale, self.dim)
                    child = np.clip(child, self.lb, self.ub)

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < self.f_opt:
                    self.f_opt = child_fitness
                    self.x_opt = child

                new_population = np.append(new_population, [child], axis=0)
                new_fitness = np.append(new_fitness, child_fitness)

            # Introduce innovators for extensive exploration
            innovators = np.random.uniform(
                self.lb, self.ub, (int(population_size * innovators_factor), self.dim)
            )
            innovator_fitness = np.array([func(ind) for ind in innovators])
            evaluations += len(innovators)

            combined_population = np.concatenate((new_population, innovators), axis=0)
            combined_fitness = np.concatenate((new_fitness, innovator_fitness), axis=0)

            indices = np.argsort(combined_fitness)[:population_size]
            population = combined_population[indices]
            fitness = combined_fitness[indices]

            mutation_scale *= adaptive_factor  # Gradually reduce mutation scale to fine-tune exploration
            if mutation_scale < 0.01:
                mutation_scale = 0.1  # Reset mutation scale if it becomes too small

        return self.f_opt, self.x_opt
