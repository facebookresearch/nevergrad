import numpy as np
from scipy.optimize import minimize


class AdaptiveDEWithElitismAndLocalSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.pop_size = 20
        self.F = 0.8  # Initial differential weight
        self.CR = 0.9  # Initial crossover probability
        self.local_search_prob = 0.1

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def nelder_mead(self, x, func):
        result = minimize(func, x, method="Nelder-Mead", bounds=[self.bounds] * self.dim)
        return result.x, result.fun

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.array([self.random_bounds() for _ in range(self.pop_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.pop_size):
                # Select three distinct individuals (but different from i)
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Adaptive differential weight and crossover probability
                F = 0.5 + np.random.rand() * 0.5
                CR = 0.5 + np.random.rand() * 0.5

                # Differential Evolution mutation and crossover
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Local Search with an adaptive probability
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

        return self.f_opt, self.x_opt
