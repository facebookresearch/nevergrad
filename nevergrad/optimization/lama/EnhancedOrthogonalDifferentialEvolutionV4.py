import numpy as np


class EnhancedOrthogonalDifferentialEvolutionV4:
    def __init__(
        self,
        budget=1000,
        population_size=50,
        mutation_factor=0.8,
        crossover_rate=0.9,
        orthogonal_factor=0.5,
        adapt_orthogonal=True,
        crossover_strategy="rand-to-best",
    ):
        self.budget = budget
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.orthogonal_factor = orthogonal_factor
        self.adapt_orthogonal = adapt_orthogonal
        self.orthogonal_factor_min = 0.1
        self.orthogonal_factor_max = 0.9
        self.orthogonal_factor_decay = 0.9
        self.crossover_strategy = crossover_strategy

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimension = len(func.bounds.lb)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))

        orthogonal_factor = self.orthogonal_factor

        for _ in range(self.budget):
            trial_population = np.zeros_like(population)

            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)
                if self.crossover_strategy == "rand-to-best":
                    rand_individual = population[np.random.choice(len(population))]
                    mutant = (
                        population[i]
                        + self.mutation_factor * (rand_individual - population[i])
                        + self.mutation_factor * (a - b)
                    )
                else:  # Default to traditional mutation
                    mutant = np.clip(a + self.mutation_factor * (b - c), func.bounds.lb, func.bounds.ub)

                orthogonal_vectors = np.random.uniform(-1, 1, size=(2, dimension))
                orthogonal_vector = np.mean(orthogonal_vectors, axis=0) * (
                    orthogonal_factor / np.sqrt(2 * np.log(dimension))
                )

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

            if self.adapt_orthogonal:
                orthogonal_factor = max(
                    orthogonal_factor * self.orthogonal_factor_decay, self.orthogonal_factor_min
                )

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.random.choice(len(population), size=3, replace=False)
        return population[idxs[0]], population[idxs[1]], population[idxs[2]]
