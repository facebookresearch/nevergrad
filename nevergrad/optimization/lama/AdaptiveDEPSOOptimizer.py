import numpy as np
from scipy.optimize import minimize


class AdaptiveDEPSOOptimizer:
    def __init__(self, budget=10000, population_size=100):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.4
        self.social_weight = 1.4

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.copy(fitness)

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                    + self.social_weight * r2 * (best_individual - population[i])
                )

                trial = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                if np.random.rand() < self.crossover_probability:
                    mutant = self.differential_mutation(population, i)
                    trial = self.differential_crossover(population[i], mutant)

                trial_fitness = evaluate(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            if eval_count < self.budget:
                elite_indices = np.argsort(fitness)[: self.population_size // 4]
                for idx in elite_indices:
                    res = self.local_search(func, population[idx])
                    eval_count += res[2]["nit"]
                    if res[1] < fitness[idx]:
                        population[idx] = res[0]
                        fitness[idx] = res[1]
                        personal_best_positions[idx] = res[0]
                        personal_best_scores[idx] = res[1]
                        if res[1] < best_fitness:
                            best_individual = res[0]
                            best_fitness = res[1]

                    if eval_count >= self.budget:
                        break

        self.f_opt = best_fitness
        self.x_opt = best_individual
        return self.f_opt, self.x_opt

    def differential_mutation(self, population, current_idx):
        indices = [idx for idx in range(self.population_size) if idx != current_idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])
        return mutant

    def differential_crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dim) < self.crossover_probability
        if not np.any(crossover_mask):
            crossover_mask[np.random.randint(0, self.dim)] = True
        trial = np.where(crossover_mask, mutant, target)
        return trial

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
