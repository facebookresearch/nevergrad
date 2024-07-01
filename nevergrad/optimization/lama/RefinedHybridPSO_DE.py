import numpy as np
from scipy.optimize import minimize


class RefinedHybridPSO_DE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.initial_pop_size = 20

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        result = minimize(func, x, method="L-BFGS-B", bounds=[self.bounds] * self.dim)
        return result.x, result.fun

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize population
        population = np.array([self.random_bounds() for _ in range(self.initial_pop_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.initial_pop_size

        # PSO parameters
        w = 0.5  # Inertia weight
        c1 = 1.5  # Cognitive coefficient
        c2 = 1.5  # Social coefficient
        velocities = np.random.uniform(-1, 1, (self.initial_pop_size, self.dim))

        while evaluations < self.budget:
            new_population = []
            new_fitness = []
            pop_size = len(population)

            for i in range(pop_size):
                # PSO update
                r1, r2 = np.random.rand(), np.random.rand()
                if self.x_opt is not None:
                    velocities[i] = (
                        w * velocities[i]
                        + c1 * r1 * (self.x_opt - population[i])
                        + c2 * r2 * (population[np.argmin(fitness)] - population[i])
                    )
                else:
                    velocities[i] = w * velocities[i] + c2 * r2 * (
                        population[np.argmin(fitness)] - population[i]
                    )

                trial_pso = population[i] + velocities[i]
                trial_pso = np.clip(trial_pso, self.bounds[0], self.bounds[1])

                # Mutation strategy from DE
                F = 0.8
                CR = 0.9
                indices = np.arange(pop_size)
                indices = np.delete(indices, i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, trial_pso)

                # Local Search
                if np.random.rand() < 0.25 and evaluations < self.budget:
                    trial, f_trial = self.local_search(trial, func)
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

            # Diversity Maintenance: Re-initialize if the population converges too tightly
            if np.std(fitness) < 1e-5 and evaluations < self.budget:
                population = np.array([self.random_bounds() for _ in range(self.initial_pop_size)])
                fitness = np.array([func(ind) for ind in population])
                evaluations += self.initial_pop_size

        return self.f_opt, self.x_opt
