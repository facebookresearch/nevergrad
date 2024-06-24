import numpy as np


class SGAE:
    def __init__(
        self, budget, population_size=100, F=0.8, CR=0.9, gradient_weight=0.1, mutation_strategy="best"
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.gradient_weight = gradient_weight  # Weight for gradient direction
        self.mutation_strategy = mutation_strategy  # Mutation strategy ('best' or 'random')

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)

            # Mutation and crossover
            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break
                # Select parents
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                x_best = population[best_idx] if self.mutation_strategy == "best" else population[a]

                # Mutation
                mutant = x_best + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Calculate gradient approximation
                perturbation = np.zeros(self.dimension)
                perturbation[np.random.randint(0, self.dimension)] = 0.01
                grad_estimated = (func(population[i] + perturbation) - fitness[i]) / 0.01
                num_evals += 1

                # Gradient exploitation
                trial -= self.gradient_weight * grad_estimated * perturbation

                trial = np.clip(trial, self.lb, self.ub)
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

            population = new_population.copy()

        return best_fitness, best_individual
