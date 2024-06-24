import numpy as np


class EnhancedAdaptiveDifferentialEvolutionWithDynamicPopulation:
    def __init__(
        self,
        budget=1000,
        init_population_size=50,
        scaling_factor_range=(0.5, 2.0),
        crossover_rate_range=(0.1, 1.0),
    ):
        self.budget = budget
        self.init_population_size = init_population_size
        self.scaling_factor_range = scaling_factor_range
        self.crossover_rate_range = crossover_rate_range

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        dimension = len(func.bounds.lb)
        population_size = self.init_population_size

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(population_size, dimension))
        fitness_values = np.array([func(ind) for ind in population])

        scaling_factors = np.random.uniform(
            self.scaling_factor_range[0], self.scaling_factor_range[1], size=population_size
        )
        crossover_rates = np.random.uniform(
            self.crossover_rate_range[0], self.crossover_rate_range[1], size=population_size
        )

        for _ in range(self.budget):
            for i in range(population_size):
                a, b, c = self.select_three_parents(population, i)

                p_best_idx = np.random.choice(np.delete(np.arange(population_size), i))
                p_best = population[p_best_idx]

                mutant = np.clip(
                    a + scaling_factors[i] * (b - c) + scaling_factors[i] * (p_best - population[i]),
                    func.bounds.lb,
                    func.bounds.ub,
                )

                crossover_points = np.random.rand(dimension) < crossover_rates[i]
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

            if np.random.rand() < 0.1:
                if population_size > 2:
                    population_size -= 1
                    idx = np.argmin(fitness_values)
                    population = np.delete(population, idx, axis=0)
                    fitness_values = np.delete(fitness_values, idx)
                    scaling_factors = np.delete(scaling_factors, idx)
                    crossover_rates = np.delete(crossover_rates, idx)
                elif population_size < self.init_population_size:
                    population_size += 1
                    new_individual = np.random.uniform(func.bounds.lb, func.bounds.ub, size=dimension)
                    population = np.vstack([population, new_individual])
                    fitness_values = np.append(fitness_values, func(new_individual))
                    scaling_factors = np.append(
                        scaling_factors, np.random.uniform(*self.scaling_factor_range)
                    )
                    crossover_rates = np.append(
                        crossover_rates, np.random.uniform(*self.crossover_rate_range)
                    )

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]

    def update_parameters(self, scaling_factors, crossover_rates, fitness_values):
        scaling_factors *= np.exp(
            0.1 * (np.mean(fitness_values) - fitness_values) / (np.std(fitness_values) + 1e-6)
        )
        crossover_rates *= np.exp(
            0.1 * (np.mean(fitness_values) - fitness_values) / (np.std(fitness_values) + 1e-6)
        )

        scaling_factors = np.clip(scaling_factors, self.scaling_factor_range[0], self.scaling_factor_range[1])
        crossover_rates = np.clip(crossover_rates, self.crossover_rate_range[0], self.crossover_rate_range[1])

        return scaling_factors, crossover_rates
