import numpy as np


class PPDE:
    def __init__(self, budget, initial_population_size=30, F_base=0.5, CR_base=0.9):
        self.budget = budget
        self.initial_population_size = initial_population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base
        self.CR_base = CR_base

    def __call__(self, func):
        # Initialize population within the bounds and compute initial fitness
        population_size = self.initial_population_size
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = population_size

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        while num_evals < self.budget:
            new_population = []
            new_fitness = []

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                # Adaptive scaling of mutation factor and crossover rate based on progress
                progress = num_evals / self.budget
                F = self.F_base * (1 + np.sin(np.pi * progress))  # Modulate F with a sine wave
                CR = self.CR_base * (0.5 + 0.5 * np.cos(np.pi * progress))  # Modulate CR with a cosine wave

                # Mutation: DE/rand/1/bin
                indices = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover: binomial
                cross_points = np.random.rand(self.dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)
            population_size = len(population)

        return best_fitness, best_individual
