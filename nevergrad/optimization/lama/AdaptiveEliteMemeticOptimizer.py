import numpy as np
from scipy.optimize import minimize


class AdaptiveEliteMemeticOptimizer:
    def __init__(self, budget=10000, population_size=50):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while eval_count < self.budget:
            new_population = []

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial[crossover_mask] = mutant[crossover_mask]
                if not np.any(crossover_mask):
                    trial[np.random.randint(0, self.dim)] = mutant[np.random.randint(0, self.dim)]

                trial_fitness = evaluate(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                else:
                    new_population.append(population[i])

                if trial_fitness < best_fitness:
                    best_individual = trial
                    best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            population = np.array(new_population)

            elite_indices = np.argsort(fitness)[: self.population_size // 4]
            for idx in elite_indices:
                res = self.local_search(func, population[idx])
                eval_count += res[2]["nit"]
                if res[1] < fitness[idx]:
                    population[idx] = res[0]
                    fitness[idx] = res[1]
                    if res[1] < best_fitness:
                        best_individual = res[0]
                        best_fitness = res[1]

                if eval_count >= self.budget:
                    break

            if eval_count < self.budget:
                population, fitness, eval_count = self.adaptive_local_refinement(
                    population, fitness, func, eval_count
                )

        self.f_opt = best_fitness
        self.x_opt = best_individual
        return self.f_opt, self.x_opt

    def local_search(self, func, x_start, tol=1e-6, max_iter=100):
        res = minimize(
            func,
            x_start,
            method="L-BFGS-B",
            bounds=[self.bounds] * self.dim,
            tol=tol,
            options={"maxiter": max_iter},
        )
        return res.x, res.fun, res

    def adaptive_local_refinement(self, population, fitness, func, eval_count):
        refined_population = []
        refined_fitness = fitness.copy()
        for i, individual in enumerate(population):
            if np.random.rand() < 0.1:  # 10% chance to refine
                res = self.local_search(func, individual)
                eval_count += res[2]["nit"]
                if res[1] < refined_fitness[i]:
                    refined_population.append(res[0])
                    refined_fitness[i] = res[1]
                else:
                    refined_population.append(individual)
            else:
                refined_population.append(individual)
            if eval_count >= self.budget:
                break
        return np.array(refined_population), refined_fitness, eval_count
