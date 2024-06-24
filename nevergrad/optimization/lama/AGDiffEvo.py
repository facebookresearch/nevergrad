import numpy as np


class AGDiffEvo:
    def __init__(self, budget, population_size=150, F_base=0.6, CR_base=0.9):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base
        self.CR_base = CR_base

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        num_evals = self.population_size

        # Track the best solution found
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Mutation parameters with Gaussian perturbation
                F = np.clip(np.random.normal(self.F_base, 0.1), 0.1, 1.0)
                CR = np.clip(np.random.normal(self.CR_base, 0.05), 0.1, 1.0)

                # Select individuals for mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(idxs, 4, replace=False)
                a, b, c, d = population[chosen]

                # Mutation: DE/current-to-best/1/bin
                mutant = population[i] + F * (best_individual - population[i]) + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            population = new_population

        return best_fitness, best_individual
