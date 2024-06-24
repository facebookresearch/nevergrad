import numpy as np


class HybridDifferentialEvolution:
    def __init__(self, budget, population_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        population = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lower_bound, upper_bound)

                # Crossover
                cross_points = np.random.rand(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            population = new_population

        return self.f_opt, self.x_opt
