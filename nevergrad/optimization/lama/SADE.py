import numpy as np


class SADE:
    def __init__(self, budget, population_size=50, F_base=0.5, CR_base=0.8):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.dimension = 5
        self.bounds = (-5.0, 5.0)  # As problem specification

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]
        evaluations = self.population_size

        # Initialize adaptive parameters
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Select mutation strategy adaptively based on fitness
                if fitness[i] < np.median(fitness):
                    F[i] *= 1.1
                else:
                    F[i] *= 0.9

                # Mutation and crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F[i] * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dimension) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                f_trial = func(trial)
                evaluations += 1

                # Selection: Accept the trial if it is better
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    CR[i] *= 1.05  # Increase CR to promote more exploration in future generations

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    CR[i] *= 0.95  # Decrease CR to reduce diversity

                # Check if budget is exhausted
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
