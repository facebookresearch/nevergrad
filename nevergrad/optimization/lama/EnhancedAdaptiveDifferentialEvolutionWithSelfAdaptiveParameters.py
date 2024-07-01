import numpy as np


class EnhancedAdaptiveDifferentialEvolutionWithSelfAdaptiveParameters:
    def __init__(self, budget=1000, population_size=50, p_best=0.2):
        self.budget = budget
        self.population_size = population_size
        self.p_best = p_best

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimension = len(func.bounds.lb)

        scaling_factor_range = (0.5, 2.0)
        crossover_rate_range = (0.1, 1.0)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))
        fitness_values = np.array([func(ind) for ind in population])

        scaling_factors = np.random.uniform(
            scaling_factor_range[0], scaling_factor_range[1], size=self.population_size
        )
        crossover_rates = np.random.uniform(
            crossover_rate_range[0], crossover_rate_range[1], size=self.population_size
        )

        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)

                p_best_idx = np.random.choice(np.delete(np.arange(self.population_size), i))
                p_best = population[p_best_idx]

                mutant = np.clip(
                    a + scaling_factors[i] * (b - c) + scaling_factors[i] * (p_best - population[i]),
                    func.bounds.lb,
                    func.bounds.ub,
                )

                crossover_points = np.random.rand(dimension) < crossover_rates[i]
                trial_individual = np.where(crossover_points, mutant, population[i])

                trial_fitness = func(trial_individual)

                if trial_fitness <= fitness_values[i]:  # Include equal fitness for diversity
                    population[i] = trial_individual
                    fitness_values[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = np.copy(trial_individual)

            # Update scaling factors and crossover rates with self-adaptation
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
        # Adapt scaling factors and crossover rates based on individuals' performance
        scaling_factors *= np.exp(
            0.1 * (np.mean(fitness_values) - fitness_values) / (np.std(fitness_values) + 1e-6)
        )
        crossover_rates *= np.exp(
            0.1 * (np.mean(fitness_values) - fitness_values) / (np.std(fitness_values) + 1e-6)
        )

        # Clip values to predefined ranges
        scaling_factors = np.clip(scaling_factors, 0.5, 2.0)
        crossover_rates = np.clip(crossover_rates, 0.1, 1.0)

        return scaling_factors, crossover_rates
