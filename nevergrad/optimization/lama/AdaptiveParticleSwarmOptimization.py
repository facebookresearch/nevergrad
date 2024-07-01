import numpy as np


class AdaptiveParticleSwarmOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        # Parameters
        self.population_size = 100
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_limit = 0.2

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        velocity = np.random.uniform(
            -self.velocity_limit, self.velocity_limit, (self.population_size, self.dim)
        )
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        personal_best_position = population.copy()
        personal_best_fitness = fitness.copy()

        evaluations = self.population_size

        while evaluations < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (evaluations / self.budget))

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocity[i] = (
                    w * velocity[i]
                    + self.c1 * r1 * (personal_best_position[i] - population[i])
                    + self.c2 * r2 * (self.x_opt - population[i])
                )
                velocity[i] = np.clip(velocity[i], -self.velocity_limit, self.velocity_limit)
                population[i] = np.clip(population[i] + velocity[i], self.lb, self.ub)

                f_candidate = func(population[i])
                evaluations += 1

                if f_candidate < personal_best_fitness[i]:
                    personal_best_position[i] = population[i].copy()
                    personal_best_fitness[i] = f_candidate

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = population[i].copy()

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
