import numpy as np


class HybridAdaptiveOrthogonalDifferentialEvolution:
    def __init__(
        self, budget=1000, population_size=50, mutation_factor=0.8, crossover_rate=0.9, orthogonal_factor=0.5
    ):
        self.budget = budget
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.orthogonal_factor = orthogonal_factor
        self.orthogonal_factor_min = 0.1
        self.orthogonal_factor_max = 0.9
        self.orthogonal_factor_decay = 0.9

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimension = len(func.bounds.lb)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))

        for _ in range(self.budget):
            trial_population = np.zeros_like(population)
            orthogonal_factor = self.orthogonal_factor

            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)

                mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)

                orthogonal_vector = np.random.normal(0, orthogonal_factor, size=dimension)

                crossover_points = np.random.rand(dimension) < self.crossover_rate
                trial_population[i] = np.where(crossover_points, mutant, population[i] + orthogonal_vector)

            trial_fitness = func(trial_population)
            population_fitness = func(population)

            improved_idxs = trial_fitness < population_fitness
            population[improved_idxs] = trial_population[improved_idxs]

            best_idx = np.argmin(trial_fitness)
            if trial_fitness[best_idx] < self.f_opt:
                self.f_opt = trial_fitness[best_idx]
                self.x_opt = trial_population[best_idx]

            orthogonal_factor = max(
                orthogonal_factor * self.orthogonal_factor_decay, self.orthogonal_factor_min
            )

            if np.random.rand() < 0.1:  # Introduce random restart
                population = np.random.uniform(
                    func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension)
                )

            # Adaptive adjustment of mutation factor and crossover rate
            self.mutation_factor = max(0.5, self.mutation_factor * 0.995)
            self.crossover_rate = min(0.95, self.crossover_rate * 1.001)

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.delete(np.arange(len(population)), current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]
