import numpy as np


class EnhancedAdaptiveDifferentialEvolutionV4:
    def __init__(self, budget=1000, population_size=50, mutation_factor=0.8, crossover_rate=0.9, p_best=0.2):
        self.budget = budget
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.p_best = p_best

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        dimension = len(func.bounds.lb)

        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.population_size, dimension))

        for _ in range(self.budget):
            for i in range(self.population_size):
                a, b, c = self.select_three_parents(population, i)

                p_best_idx = np.random.choice(np.delete(np.arange(self.population_size), i))
                p_best = population[p_best_idx]

                mutant = np.clip(
                    a + self.mutation_factor * (b - c) + self.mutation_factor * (p_best - population[i]),
                    func.bounds.lb,
                    func.bounds.ub,
                )

                crossover_points = np.random.rand(dimension) < self.crossover_rate
                trial_individual = np.where(crossover_points, mutant, population[i])

                trial_fitness = func(trial_individual)
                current_fitness = func(population[i])

                if trial_fitness < current_fitness:
                    population[i] = trial_individual
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = np.copy(trial_individual)

        return self.f_opt, self.x_opt

    def select_three_parents(self, population, current_idx):
        idxs = np.arange(len(population))
        idxs = np.delete(idxs, current_idx)
        selected_idxs = np.random.choice(idxs, size=3, replace=False)
        return population[selected_idxs[0]], population[selected_idxs[1]], population[selected_idxs[2]]
