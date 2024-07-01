import numpy as np


class CovarianceMatrixAdaptationDifferentialEvolution:
    def __init__(self, budget, population_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F = F
        self.CR = CR

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

                # Adaptive Mutation and Crossover
                self.F = 0.5 + 0.3 * np.random.rand()
                self.CR = 0.8 + 0.2 * np.random.rand()

                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lower_bound, upper_bound)

                cross_points = np.random.rand(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Covariance Matrix Adaptation
            mean = np.mean(new_population, axis=0)
            cov_matrix = np.cov(new_population.T)
            cov_matrix = (cov_matrix + cov_matrix.T) / 2  # Ensure symmetry
            cov_matrix = np.clip(cov_matrix, -1, 1)  # Prevent numerical issues

            population = np.random.multivariate_normal(mean, cov_matrix, self.population_size)
            population = np.clip(population, lower_bound, upper_bound)
            fitness = np.array([func(ind) for ind in population])

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
