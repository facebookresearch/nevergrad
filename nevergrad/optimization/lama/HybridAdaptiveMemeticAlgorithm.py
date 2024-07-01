import numpy as np


class HybridAdaptiveMemeticAlgorithm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def local_search(self, x, func):
        """Simple local search around a point"""
        best_x = x
        best_f = func(x)
        for i in range(10):
            perturbation = np.random.uniform(-0.1, 0.1, self.dim)
            new_x = np.clip(x + perturbation, self.lb, self.ub)
            new_f = func(new_x)
            if new_f < best_f:
                best_x = new_x
                best_f = new_f
        return best_x, best_f

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population_size = 50
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = len(fitness)

        # Initialize strategy parameters for each individual
        F = np.random.uniform(0.5, 1.0, population_size)
        CR = np.random.uniform(0.1, 0.9, population_size)

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation and Crossover using Differential Evolution
                indices = np.random.choice([j for j in range(population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = x1 + F[i] * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                trial_vector = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < CR[i]
                if not np.any(crossover_points):
                    crossover_points[np.random.randint(0, self.dim)] = True
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                # Evaluate trial vector
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Adapt strategy parameters
                    F[i] = F[i] + 0.1 * (np.random.rand() - 0.5)
                    F[i] = np.clip(F[i], 0.5, 1.0)
                    CR[i] = CR[i] + 0.1 * (np.random.rand() - 0.5)
                    CR[i] = np.clip(CR[i], 0.1, 0.9)

                    # Update global best
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

                # Apply local search on some individuals
                if np.random.rand() < 0.1:
                    local_best_x, local_best_f = self.local_search(population[i], func)
                    evaluations += 10  # Assuming local search uses 10 evaluations
                    if local_best_f < fitness[i]:
                        population[i] = local_best_x
                        fitness[i] = local_best_f
                        if local_best_f < self.f_opt:
                            self.f_opt = local_best_f
                            self.x_opt = local_best_x

            # Periodically introduce new random solutions (to avoid local optima)
            if evaluations % (population_size // 2) == 0:
                new_population = np.random.uniform(self.lb, self.ub, (population_size // 5, self.dim))
                new_fitness = np.array([func(ind) for ind in new_population])
                evaluations += len(new_fitness)

                # Replace worst individuals with new random individuals
                worst_indices = fitness.argsort()[-(population_size // 5) :]
                population[worst_indices] = new_population
                fitness[worst_indices] = new_fitness

                # Reinitialize strategy parameters for new individuals
                F[worst_indices] = np.random.uniform(0.5, 1.0, population_size // 5)
                CR[worst_indices] = np.random.uniform(0.1, 0.9, population_size // 5)

        return self.f_opt, self.x_opt
