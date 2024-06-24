import numpy as np
from scipy.optimize import minimize


class AdaptiveHybridDEWithIntensifiedLocalSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.initial_pop_size = 20
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.local_search_prob = 0.2  # Increased local search probability

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def nelder_mead(self, x, func):
        result = minimize(func, x, method="Nelder-Mead", bounds=[self.bounds] * self.dim)
        return result.x, result.fun

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.array([self.random_bounds() for _ in range(self.initial_pop_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_pop_size

        while evaluations < self.budget:
            new_population = []
            new_fitness = []
            pop_size = len(population)

            for i in range(pop_size):
                # Select mutation strategy adaptively
                strategy = np.random.choice(["rand/1", "best/1"])

                if strategy == "rand/1":
                    # Select three distinct individuals (but different from i)
                    indices = np.arange(pop_size)
                    indices = indices[indices != i]
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                elif strategy == "best/1":
                    best_idx = np.argmin(fitness)
                    best = population[best_idx]
                    indices = np.arange(pop_size)
                    indices = indices[indices != best_idx]
                    b, c = population[np.random.choice(indices, 2, replace=False)]
                    mutant = np.clip(best + self.F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Local Search with an increased probability
                if np.random.rand() < self.local_search_prob and evaluations + 1 <= self.budget:
                    trial, f_trial = self.nelder_mead(trial, func)
                    evaluations += 1
                else:
                    f_trial = func(trial)
                    evaluations += 1

                # Selection
                if f_trial < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

                # Check if we've exhausted our budget
                if evaluations >= self.budget:
                    break

            # Elitism: Keep the best individual
            best_idx = np.argmin(new_fitness)
            best_individual = new_population[best_idx]
            best_fitness = new_fitness[best_idx]
            if best_fitness < self.f_opt:
                self.f_opt = best_fitness
                self.x_opt = best_individual

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            # Adjust population size based on convergence
            if np.std(fitness) < 1e-5:
                if len(population) > 10:
                    population = population[: len(population) // 2]
                    fitness = fitness[: len(fitness) // 2]
            else:
                if (
                    len(population) < self.initial_pop_size * 2
                    and evaluations + len(population) <= self.budget
                ):
                    new_individuals = np.array([self.random_bounds() for _ in range(len(population))])
                    new_fitnesses = np.array([func(ind) for ind in new_individuals])
                    population = np.vstack((population, new_individuals))
                    fitness = np.hstack((fitness, new_fitnesses))
                    evaluations += len(new_individuals)

        return self.f_opt, self.x_opt
