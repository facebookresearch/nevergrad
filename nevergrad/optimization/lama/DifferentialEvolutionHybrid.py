import numpy as np


class DifferentialEvolutionHybrid:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0
        self.mutation_factor = 0.8
        self.crossover_prob = 0.7

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population_size = 20
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = len(fitness)

        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant_vector = x1 + self.mutation_factor * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                # Crossover
                trial_vector = np.copy(population[i])
                crossover_points = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover_points] = mutant_vector[crossover_points]

                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update global best
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

        return self.f_opt, self.x_opt
