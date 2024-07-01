import numpy as np


class AdaptiveMemeticOptimizerV2:
    def __init__(self, budget=10000, population_size=40):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        eval_count = self.population_size

        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best = population[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_coeff * r1 * (personal_best[i] - population[i])
                social_velocity = self.social_coeff * r2 * (global_best - population[i])
                velocity[i] = self.inertia_weight * velocity[i] + cognitive_velocity + social_velocity
                population[i] = np.clip(population[i] + velocity[i], self.bounds[0], self.bounds[1])

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_probability
                trial[crossover_mask] = mutant[crossover_mask]

                trial_fitness = func(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < personal_best_fitness[i]:
                        personal_best[i] = trial
                        personal_best_fitness[i] = trial_fitness

                        if trial_fitness < global_best_fitness:
                            global_best = trial
                            global_best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            self.inertia_weight = max(0.4, self.inertia_weight * 0.98)
            self.mutation_factor = np.random.uniform(0.6, 0.9)
            self.crossover_probability = np.random.uniform(0.7, 0.95)
            self.cognitive_coeff = np.random.uniform(1.3, 1.7)
            self.social_coeff = np.random.uniform(1.3, 1.7)

            if eval_count >= self.budget:
                break

            # Enhanced Local Search on selected individuals
            if eval_count + self.population_size / 2 <= self.budget:
                for i in np.random.choice(self.population_size, self.population_size // 2, replace=False):
                    res = self.nelder_mead(func, population[i])
                    if res[1] < fitness[i]:
                        population[i] = res[0]
                        fitness[i] = res[1]
                        if res[1] < global_best_fitness:
                            global_best = res[0]
                            global_best_fitness = res[1]

        self.f_opt = global_best_fitness
        self.x_opt = global_best
        return self.f_opt, self.x_opt

    def nelder_mead(self, func, x_start, tol=1e-6, max_iter=100):
        from scipy.optimize import minimize

        res = minimize(func, x_start, method="Nelder-Mead", tol=tol, options={"maxiter": max_iter})
        return res.x, res.fun
