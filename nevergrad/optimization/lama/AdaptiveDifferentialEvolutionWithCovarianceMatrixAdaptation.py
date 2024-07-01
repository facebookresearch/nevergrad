import numpy as np


class AdaptiveDifferentialEvolutionWithCovarianceMatrixAdaptation:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        bounds = np.array([-5.0, 5.0])
        population_size = 20
        F = 0.8  # Initial differential weight
        CR = 0.9  # Initial crossover probability

        # Initialize population
        population = np.random.uniform(bounds[0], bounds[1], (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evaluations = population_size

        # Initialize self-adaptive parameters
        F_values = np.full(population_size, F)
        CR_values = np.full(population_size, CR)

        while evaluations < self.budget:
            # Covariance Matrix Adaptation
            mean = np.mean(population, axis=0)
            cov_matrix = np.cov(population.T)
            cov_matrix = (cov_matrix + cov_matrix.T) / 2 + np.eye(
                self.dim
            ) * 1e-6  # Ensure positive semi-definiteness

            new_population = np.zeros_like(population)
            new_fitness = np.zeros(population_size)
            new_F_values = np.zeros(population_size)
            new_CR_values = np.zeros(population_size)

            for i in range(population_size):
                # Adaptation of F and CR
                if np.random.rand() < 0.1:
                    F_values[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR_values[i] = np.random.rand()

                # Mutation
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F_values[i] * (b - c), bounds[0], bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR_values[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = f_trial
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                    # Update the best found solution
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    new_F_values[i] = F_values[i]
                    new_CR_values[i] = CR_values[i]

                if evaluations >= self.budget:
                    break

            population, fitness = new_population, new_fitness
            F_values, CR_values = new_F_values, new_CR_values

            # Apply Covariance Matrix Adaptation
            covariance_update_population = np.zeros_like(population)
            for i in range(population_size):
                perturbation = np.random.multivariate_normal(np.zeros(self.dim), cov_matrix)
                covariance_update_population[i] = np.clip(mean + perturbation, bounds[0], bounds[1])
                f_trial = func(covariance_update_population[i])
                evaluations += 1

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = covariance_update_population[i]

                if evaluations >= self.budget:
                    break

            population = np.concatenate((population, covariance_update_population), axis=0)
            fitness = np.array([func(ind) for ind in population])
            best_indices = np.argsort(fitness)[:population_size]
            population = population[best_indices]
            fitness = fitness[best_indices]

        return self.f_opt, self.x_opt
