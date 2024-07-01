import numpy as np


class RPWDE:
    def __init__(self, budget, population_size=50, crossover_rate=0.9, F_base=0.5, F_amp=0.4, elite_size=5):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population within the bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Elite solutions tracking
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite solutions
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices]
            elite_fitness = fitness[elite_indices]

            for i in range(self.population_size):
                # Adaptive mutation factor with progressive wave pattern
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Select mutation candidates
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                best_or_elite = elite[np.random.randint(0, self.elite_size)]

                # Mutation: A combination of best or elite with random individuals
                mutant = np.clip(best_or_elite + F * (a - b), lb, ub)

                # Crossover: Binomial
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
