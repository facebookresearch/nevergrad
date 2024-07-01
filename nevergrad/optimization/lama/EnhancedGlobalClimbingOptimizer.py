import numpy as np


class EnhancedGlobalClimbingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 300  # Adjusted population size for a balance between exploration and convergence
        elite_size = 50  # Adjusted elite size to maintain quality without hindering diversity
        evaluations = 0

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        mutation_scale = 0.1  # Fine-tuned mutation scale
        adaptive_factor = 0.90  # Enhanced adaptiveness to fitness landscape changes
        recombination_prob = 0.8  # Tuned recombination probability for stability

        # Evolution loop
        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(population_size):
                if np.random.rand() < recombination_prob:
                    # Differential evolution inspired crossover strategy
                    indices = np.random.choice(population_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    child = x0 + mutation_scale * (x1 - x2)  # Differential mutation
                    child = np.clip(child, self.lb, self.ub)
                else:
                    # Mutation and selection
                    child = population[i] + np.random.normal(0, mutation_scale, self.dim)
                    child = np.clip(child, self.lb, self.ub)

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < fitness[i]:
                    new_population.append(child)
                    new_fitness.append(child_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < self.f_opt:
                self.f_opt = fitness[current_best_idx]
                self.x_opt = population[current_best_idx]

            # Adaptive mutation scaling and replacement strategy
            if evaluations % 400 == 0:
                mutation_scale *= adaptive_factor  # Decrement to stabilize as optimization progresses
                elite_indices = np.argsort(fitness)[:elite_size]
                elite_individuals = population[elite_indices]
                for idx in range(population_size - elite_size):
                    if np.random.rand() < 0.2:  # Introducing fresh blood
                        replacement_idx = np.random.choice(elite_size)
                        population[idx] = elite_individuals[replacement_idx]
                        fitness[idx] = func(population[idx])
                        evaluations += 1

        return self.f_opt, self.x_opt
