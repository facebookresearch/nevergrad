import numpy as np


class ExDADe:
    def __init__(self, budget, population_size=20, F_base=0.8, CR_base=0.7, epsilon=1e-10):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base
        self.CR_base = CR_base
        self.epsilon = epsilon

    def __call__(self, func):
        # Initialize population within the bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        num_evals = self.population_size

        # Tracking the best solution found
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Continued adaptation mechanism for mutation and crossover parameters
        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Enhanced adaptation for F and CR based on progress and diversity
                progress = num_evals / self.budget
                diversity = np.std(population).max() + self.epsilon
                F = self.F_base * (0.5 + np.random.rand()) * diversity
                CR = self.CR_base * (1 - np.exp(-3 * progress))

                # Mutation: "current-to-best/1"
                j_rand = np.random.randint(self.dimension)
                mutant = (
                    population[i]
                    + F * (best_individual - population[i])
                    + F
                    * (
                        population[np.random.randint(self.population_size)]
                        - population[np.random.randint(self.population_size)]
                    )
                )
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover: binomial
                trial = np.where(np.random.rand(self.dimension) < CR, mutant, population[i])
                trial[j_rand] = mutant[j_rand]  # Ensuring at least one dimension comes from mutant

                # Evaluate the trial solution
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
