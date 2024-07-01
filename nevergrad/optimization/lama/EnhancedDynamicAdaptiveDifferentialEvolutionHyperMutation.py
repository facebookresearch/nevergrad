import numpy as np


class EnhancedDynamicAdaptiveDifferentialEvolutionHyperMutation:
    def __init__(
        self,
        budget=1000,
        population_size=50,
        scaling_factor_range=(0.1, 1.0),
        crossover_rate_range=(0.1, 1.0),
        hypermutation_probability=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.scaling_factor_range = scaling_factor_range
        self.crossover_rate_range = crossover_rate_range
        self.hypermutation_probability = hypermutation_probability

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

                trial_individual = self.generate_trial_individual(
                    population[i], a, b, c, p_best, scaling_factor, crossover_rate, func
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

            if np.random.rand() < self.hypermutation_probability:
                population = self.hypermutate_population(population, func)

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]

    def generate_trial_individual(self, current, a, b, c, p_best, scaling_factor, crossover_rate, func):
        dimension = len(current)
        mutant = np.clip(
            a + scaling_factor * (b - c) + scaling_factor * (p_best - current), func.bounds.lb, func.bounds.ub
        )
        crossover_points = np.random.rand(dimension) < crossover_rate
        return np.where(crossover_points, mutant, current)

    def update_parameters(self, scaling_factors, crossover_rates, fitness_values):
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        scaling_factor_adaptation = 0.1 * (mean_fitness - fitness_values) / (std_fitness + 1e-6) + 0.05 * (
            fitness_values.min() - fitness_values
        )
        crossover_rate_adaptation = 0.1 * (mean_fitness - fitness_values) / (std_fitness + 1e-6) + 0.05 * (
            fitness_values.min() - fitness_values
        )

        new_scaling_factors = scaling_factors * np.exp(scaling_factor_adaptation)
        new_crossover_rates = crossover_rates * np.exp(crossover_rate_adaptation)

        return np.clip(new_scaling_factors, *self.scaling_factor_range), np.clip(
            new_crossover_rates, *self.crossover_rate_range
        )

    def hypermutate_population(self, population, func):
        dimension = len(population[0])
        mutated_population = population + np.random.normal(0, 0.1, size=(self.population_size, dimension))
        mutated_population = np.clip(mutated_population, func.bounds.lb, func.bounds.ub)
        return mutated_population
