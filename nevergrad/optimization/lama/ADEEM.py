import numpy as np


class ADEEM:
    def __init__(self, budget, population_size=50, F=0.8, CR=0.9, alpha=0.1):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.alpha = alpha  # Rate of adaptive adjustment

    def __call__(self, func):
        # Initialize population and fitness
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Mutation strategy using "DE/rand/1/bin"
                perm = np.random.permutation(self.population_size)
                perm = perm[perm != i][:3]

                a, b, c = population[perm[0]], population[perm[1]], population[perm[2]]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            # Adaptation of F and CR using feedback from the current population
            mean_fitness = np.mean(fitness)
            self.F = np.clip(self.F * (1 + self.alpha * (best_fitness - mean_fitness)), 0.5, 1)
            self.CR = np.clip(self.CR * (1 - self.alpha * (best_fitness - mean_fitness)), 0.6, 0.95)

            population = new_population.copy()

        return best_fitness, best_individual
