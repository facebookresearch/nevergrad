import numpy as np


class DualModeOptimization:
    def __init__(self, budget, dimension=5, population_size=20, mutation_scale=0.1, gradient_intensity=5):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.mutation_scale = mutation_scale
        self.gradient_intensity = gradient_intensity  # Intensity of gradient-based local search

    def __call__(self, func):
        # Initialize population within bounds [-5.0, 5.0]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        best_idx = np.argmin(fitness)
        f_opt = fitness[best_idx]
        x_opt = population[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Tournament selection for mutation
            for i in range(self.population_size):
                candidates_idx = np.random.choice(self.population_size, 3, replace=False)
                candidates_fitness = fitness[candidates_idx]
                best_local_idx = np.argmin(candidates_fitness)
                target_idx = candidates_idx[best_local_idx]

                # Mutation using differential evolution strategy
                r1, r2, r3 = np.random.choice(self.population_size, 3, replace=False)
                mutant = population[r1] + self.mutation_scale * (population[r2] - population[r3])
                mutant = np.clip(mutant, -5.0, 5.0)  # Ensure mutant is within bounds

                # Evaluate mutant
                mutant_fitness = func(mutant)
                evaluations += 1

                # Replace if mutant is better
                if mutant_fitness < fitness[target_idx]:
                    population[target_idx] = mutant
                    fitness[target_idx] = mutant_fitness

                # Update global optimum
                if mutant_fitness < f_opt:
                    f_opt = mutant_fitness
                    x_opt = mutant

                if evaluations >= self.budget:
                    break

            # Perform gradient-based local search on the best solution
            if evaluations + self.gradient_intensity <= self.budget:
                local_opt = x_opt.copy()
                for _ in range(self.gradient_intensity):
                    gradient = np.array(
                        [
                            (func(local_opt + eps * np.eye(1, self.dimension, k)[0]) - func(local_opt)) / eps
                            for k, eps in enumerate([1e-5] * self.dimension)
                        ]
                    )
                    local_opt -= 0.01 * gradient  # Small gradient step
                    local_opt = np.clip(local_opt, -5.0, 5.0)
                    local_fitness = func(local_opt)
                    evaluations += 1

                    if local_fitness < f_opt:
                        f_opt = local_fitness
                        x_opt = local_opt
                    else:
                        break

                if evaluations >= self.budget:
                    break

        return f_opt, x_opt
