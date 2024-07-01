import numpy as np


class EnhancedDifferentialEvolutionAdaptivePSO:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        # Parameters for PSO
        self.population_size = 100
        self.w_min = 0.4
        self.w_max = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.velocity_limit = 0.2

        # Parameters for DE
        self.F = 0.8
        self.CR = 0.9

        # Parameters for adaptive DE
        self.F_l = 0.5
        self.F_u = 1.0
        self.CR_l = 0.1
        self.CR_u = 0.9

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
                # PSO Update
                r1, r2 = np.random.rand(2)
                velocity[i] = (
                    w * velocity[i]
                    + self.c1 * r1 * (personal_best_position[i] - population[i])
                    + self.c2 * r2 * (self.x_opt - population[i])
                )

                # Adaptive velocity clamping
                velocity_magnitude = np.linalg.norm(velocity[i])
                if velocity_magnitude > self.velocity_limit:
                    velocity[i] = (velocity[i] / velocity_magnitude) * self.velocity_limit

                new_position = np.clip(population[i] + velocity[i], self.lb, self.ub)

                # Adaptive DE Update
                if np.random.rand() < 0.5:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    F_adaptive = self.F_l + np.random.rand() * (self.F_u - self.F_l)
                    CR_adaptive = self.CR_l + np.random.rand() * (self.CR_u - self.CR_l)
                    mutant_vector = np.clip(a + F_adaptive * (b - c), self.lb, self.ub)

                    trial_vector = np.copy(population[i])
                    for j in range(self.dim):
                        if np.random.rand() < CR_adaptive:
                            trial_vector[j] = mutant_vector[j]

                    new_position = trial_vector

                f_candidate = func(new_position)
                evaluations += 1

                if f_candidate < personal_best_fitness[i]:
                    personal_best_position[i] = new_position.copy()
                    personal_best_fitness[i] = f_candidate

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = new_position.copy()

                population[i] = new_position

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
