import numpy as np


class EnhancedDifferentiatedAdaptiveEvolution:
    def __init__(
        self,
        budget=1000,
        population_size=50,
        mutation_factor=(0.5, 2.0),
        crossover_rate=(0.1, 1.0),
        p_best=0.2,
        scaling_factor=0.5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.p_best = p_best
        self.scaling_factor = scaling_factor

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimension = len(func.bounds.lb)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))
        fitness_values = np.array([func(ind) for ind in population])

        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)

                p_best_idx = np.random.choice(np.delete(np.arange(self.population_size), i))
                p_best = population[p_best_idx]

                mutation_factor = self.adapt_parameter(self.mutation_factor, fitness_values, i)
                crossover_rate = self.adapt_parameter(self.crossover_rate, fitness_values, i)

                mutant = np.clip(
                    a + mutation_factor * (b - c) + mutation_factor * (p_best - population[i]),
                    func.bounds.lb,
                    func.bounds.ub,
                )

                crossover_points = np.random.rand(dimension) < crossover_rate
                trial_individual = np.where(crossover_points, mutant, population[i])

                trial_fitness = func(trial_individual)

                if trial_fitness <= fitness_values[i]:  # Include equal fitness for diversity
                    population[i] = trial_individual
                    fitness_values[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = np.copy(trial_individual)

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]

    def adapt_parameter(self, parameter_range, fitness_values, idx):
        sorted_fitness_idxs = np.argsort(fitness_values)
        best_idx = sorted_fitness_idxs[0]

        if idx == best_idx:
            return parameter_range[1]

        worst_idx = sorted_fitness_idxs[-1]
        diff = np.abs(fitness_values[best_idx] - fitness_values[worst_idx])

        if diff == 0:
            return parameter_range[0]

        return np.clip(
            parameter_range[0]
            + (parameter_range[1] - parameter_range[0])
            * (fitness_values[idx] - fitness_values[worst_idx])
            / diff,
            parameter_range[0],
            parameter_range[1],
        )
