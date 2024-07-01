import numpy as np
from scipy.optimize import minimize


class HybridDifferentialEvolutionMemeticOptimizer:
    def __init__(self, budget=10000, population_size=100):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.elite_fraction = 0.2
        self.local_search_probability = 0.3
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.memory_size = 30

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        performance_memory = [best_fitness] * self.memory_size

        while eval_count < self.budget:
            new_population = []
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = evaluate(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])

                if eval_count >= self.budget:
                    break

            population = np.array(new_population)

            # Local search on elite individuals
            elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
            for idx in elite_indices:
                if np.random.rand() < self.local_search_probability:
                    res = self.local_search(func, population[idx])
                    if res is not None:
                        eval_count += 1
                        if res[1] < fitness[idx]:
                            population[idx] = res[0]
                            fitness[idx] = res[1]
                            if res[1] < best_fitness:
                                best_individual = res[0]
                                best_fitness = res[1]

            performance_memory.append(best_fitness)
            if len(performance_memory) > self.memory_size:
                performance_memory.pop(0)

            if eval_count >= self.budget:
                break

        self.f_opt = best_fitness
        self.x_opt = best_individual
        return self.f_opt, self.x_opt

    def local_search(self, func, x_start, tol=1e-6, max_iter=50):
        res = minimize(
            func,
            x_start,
            method="L-BFGS-B",
            bounds=[self.bounds] * self.dim,
            tol=tol,
            options={"maxiter": max_iter},
        )
        if res.success:
            return res.x, res.fun
        return None


# Example usage
# optimizer = HybridDifferentialEvolutionMemeticOptimizer(budget=10000)
# best_fitness, best_solution = optimizer(some_black_box_function)
