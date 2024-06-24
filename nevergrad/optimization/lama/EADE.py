import numpy as np


class EADe:
    def __init__(self, budget, population_size=30, F_base=0.6, CR_base=0.9):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base
        self.CR_base = CR_base

    def __call__(self, func):
        # Initialize population within the bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        num_evals = self.population_size

        # Tracking the best solution found
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Begin the evolutionary process
        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Jitter and Amplify the F and CR according to the stage of optimization
                F = self.F_base + 0.1 * np.sin((num_evals / self.budget) * np.pi)
                CR = self.CR_base * (0.5 + 0.5 * np.cos((num_evals / self.budget) * np.pi))

                # Mutation using differential evolution strategy "rand/1/bin"
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial solution
                trial_fitness = func(trial)
                num_evals += 1

                # Selection: Greedily select the better vector
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            # Update the population with new generation
            population = new_population

        return best_fitness, best_individual
