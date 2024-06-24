import numpy as np


class EnhancedOrthogonalDifferentialEvolution:
    def __init__(self, budget=1000, population_size=50, mutation_factor=0.8, crossover_rate=0.9):
        self.budget = budget
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimension = len(func.bounds.lb)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))

        for _ in range(self.budget):
            trial_population = np.zeros_like(population)

            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)
                mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)

                # Introducing Enhanced Orthogonal Crossover
                orthogonal_vectors = np.random.uniform(-1, 1, size=(2, dimension))
                orthogonal_vector = np.mean(orthogonal_vectors, axis=0)
                crossover_points = np.random.rand(dimension) < self.crossover_rate
                trial_population[i] = np.where(crossover_points, mutant, population[i])

            trial_fitness = func(trial_population)
            population_fitness = func(population)

            improved_idxs = trial_fitness < population_fitness
            population[improved_idxs] = trial_population[improved_idxs]

            best_idx = np.argmin(trial_fitness)
            if trial_fitness[best_idx] < self.f_opt:
                self.f_opt = trial_fitness[best_idx]
                self.x_opt = trial_population[best_idx]

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.random.choice(len(population), size=3, replace=False)
        return population[idxs[0]], population[idxs[1]], population[idxs[2]]
