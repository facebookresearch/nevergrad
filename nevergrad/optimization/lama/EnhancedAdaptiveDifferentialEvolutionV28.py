import numpy as np


class EnhancedAdaptiveDifferentialEvolutionV28:
    def __init__(
        self,
        budget=1000,
        population_size=50,
        scaling_factor_range=(0.5, 0.9),
        crossover_rate_range=(0.6, 1.0),
        diversification_factor=0.1,
        dynamic_step=0.2,
    ):
        self.budget = budget
        self.population_size = population_size
        self.scaling_factor_range = scaling_factor_range
        self.crossover_rate_range = crossover_rate_range
        self.diversification_factor = diversification_factor
        self.dynamic_step = dynamic_step

    def __call__(self, func):
        self.func = func
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

                scaling_factor = scaling_factors[i]
                crossover_rate = crossover_rates[i]

                trial_individual = self.generate_trial_individual(
                    population[i], a, b, c, scaling_factor, crossover_rate
                )

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

            population = self.population_diversification(population)

            # Adjust parameters dynamically
            scaling_factor_range = np.clip(
                np.array(self.scaling_factor_range)
                * (1 + self.dynamic_step * (np.mean(fitness_values) - np.min(fitness_values))),
                0.5,
                0.9,
            )
            crossover_rate_range = np.clip(
                np.array(self.crossover_rate_range)
                * (1 + self.dynamic_step * (np.mean(fitness_values) - np.min(fitness_values))),
                0.6,
                1.0,
            )

            scaling_factors = np.clip(scaling_factors, *scaling_factor_range)
            crossover_rates = np.clip(crossover_rates, *crossover_rate_range)

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]

    def generate_trial_individual(self, current, a, b, c, scaling_factor, crossover_rate):
        dimension = len(current)
        mutant = np.clip(a + scaling_factor * (b - c), self.func.bounds.lb, self.func.bounds.ub)
        crossover_points = np.random.rand(dimension) < crossover_rate
        return np.where(crossover_points, mutant, current)

    def update_parameters(self, scaling_factors, crossover_rates, fitness_values):
        scaling_factor_range = np.clip(
            np.array(self.scaling_factor_range) * (1 + 0.1 * np.mean(fitness_values)), 0.5, 0.9
        )
        crossover_rate_range = np.clip(
            np.array(self.crossover_rate_range) * (1 + 0.1 * np.mean(fitness_values)), 0.6, 1.0
        )

        return np.clip(scaling_factors, *scaling_factor_range), np.clip(
            crossover_rates, *crossover_rate_range
        )

    def population_diversification(self, population):
        mean_individual = np.mean(population, axis=0)
        std_individual = np.std(population, axis=0)
        diversity_index = np.sum(std_individual) / len(std_individual)

        if diversity_index < self.diversification_factor:
            mutated_population = np.clip(
                population + np.random.normal(0, 0.1, size=population.shape),
                self.func.bounds.lb,
                self.func.bounds.ub,
            )
            return mutated_population
        else:
            return population
