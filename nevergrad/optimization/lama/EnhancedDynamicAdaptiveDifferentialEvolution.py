import numpy as np


class EnhancedDynamicAdaptiveDifferentialEvolution:
    def __init__(
        self,
        budget=1000,
        population_size=50,
        scaling_factor_range=(0.1, 1.0),
        crossover_rate_range=(0.1, 1.0),
    ):
        self.budget = budget
        self.population_size = population_size
        self.scaling_factor_range = scaling_factor_range
        self.crossover_rate_range = crossover_rate_range

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        dimension = len(func.bounds.lb)
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))
        fitness_values = np.array([func(ind) for ind in population])

        scaling_factors = np.full(self.population_size, np.mean(self.scaling_factor_range))
        crossover_rates = np.full(self.population_size, np.mean(self.crossover_rate_range))

        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)

                p_best_idx = np.random.choice(np.delete(np.arange(self.population_size), i))
                p_best = population[p_best_idx]

                scaling_factor = scaling_factors[i]
                crossover_rate = crossover_rates[i]

                mutant = np.clip(
                    a + scaling_factor * (b - c) + scaling_factor * (p_best - population[i]),
                    func.bounds.lb,
                    func.bounds.ub,
                )

                crossover_points = np.random.rand(dimension) < crossover_rate
                trial_individual = np.where(crossover_points, mutant, population[i])

                trial_fitness = func(trial_individual)

                if trial_fitness <= fitness_values[i]:
                    population[i] = trial_individual
                    fitness_values[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = np.copy(trial_individual)

            scaling_factors, crossover_rates = self.update_parameters(
                scaling_factors, crossover_rates, fitness_values
            )

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]

    def update_parameters(self, scaling_factors, crossover_rates, fitness_values):
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        new_scaling_factors = scaling_factors * np.exp(
            0.05 * (mean_fitness - fitness_values) / (std_fitness + 1e-6)
            + 0.05 * (fitness_values.min() - fitness_values)
        )
        new_crossover_rates = crossover_rates * np.exp(
            0.05 * (mean_fitness - fitness_values) / (std_fitness + 1e-6)
            + 0.05 * (fitness_values.min() - fitness_values)
        )

        return np.clip(new_scaling_factors, *self.scaling_factor_range), np.clip(
            new_crossover_rates, *self.crossover_rate_range
        )
