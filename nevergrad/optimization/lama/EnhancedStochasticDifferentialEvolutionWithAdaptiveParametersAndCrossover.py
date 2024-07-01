import numpy as np


class EnhancedStochasticDifferentialEvolutionWithAdaptiveParametersAndCrossover:
    def __init__(
        self,
        budget=1000,
        population_size=50,
        diversification_factor=0.1,
        cr_range=(0.2, 0.9),
        f_range=(0.2, 0.8),
    ):
        self.budget = budget
        self.population_size = population_size
        self.diversification_factor = diversification_factor
        self.cr_range = cr_range
        self.f_range = f_range

    def __call__(self, func):
        self.func = func
        self.f_opt = np.inf
        self.x_opt = None

        dimension = len(func.bounds.lb)
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))
        fitness_values = np.array([func(ind) for ind in population])

        cr = np.random.uniform(*self.cr_range, size=self.population_size)
        f = np.random.uniform(*self.f_range, size=self.population_size)

        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)
                trial_individual = self.generate_trial_individual(population[i], a, b, c, f[i], cr[i])

                trial_fitness = func(trial_individual)

                if trial_fitness <= fitness_values[i]:
                    population[i] = trial_individual
                    fitness_values[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = np.copy(trial_individual)

            cr, f = self.adapt_parameters(cr, f, fitness_values)

            population = self.population_diversification(population)

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]

    def generate_trial_individual(self, current, a, b, c, f, cr):
        dimension = len(current)
        mutant = np.clip(a + f * (b - c), self.func.bounds.lb, self.func.bounds.ub)
        crossover_points = np.random.rand(dimension) < cr
        return np.where(crossover_points, mutant, current)

    def adapt_parameters(self, cr, f, fitness_values):
        mean_fitness = np.mean(fitness_values)
        cr = cr * (1 + 0.1 * (mean_fitness - fitness_values))
        f = f * (1 + 0.1 * (mean_fitness - fitness_values))
        return np.clip(cr, *self.cr_range), np.clip(f, *self.f_range)

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
