import numpy as np


class ADEPM:
    def __init__(self, budget, population_size=30, F_mean=0.5, CR_mean=0.9, learning_rate=0.1):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_mean = F_mean  # Mean differential weight
        self.CR_mean = CR_mean  # Mean crossover probability
        self.learning_rate = learning_rate  # Learning rate for adaptation

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main evolutionary loop
        while num_evals < self.budget:
            # Mutation and crossover parameters adaptively updated
            F = np.clip(np.random.normal(self.F_mean, 0.1), 0.1, 1)
            CR = np.clip(np.random.normal(self.CR_mean, 0.1), 0.1, 1)

            for i in range(self.population_size):
                # Mutation
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial vector
                trial_fitness = func(trial)
                num_evals += 1
                if num_evals >= self.budget:
                    break

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

            # Adapt F and CR means
            self.F_mean += self.learning_rate * (F - self.F_mean)
            self.CR_mean += self.learning_rate * (CR - self.CR_mean)

        return best_fitness, best_individual
