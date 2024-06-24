import numpy as np
from scipy.optimize import minimize


class AdaptiveQuantumPSO:
    def __init__(self, budget=10000, population_size=100):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.inertia_weight = 0.729
        self.cognitive_weight = 1.49445
        self.social_weight = 1.49445
        self.quantum_weight = 0.3  # parameter for quantum behavior
        self.adaptive_threshold = 0.1  # threshold for triggering adaptive behavior
        self.elite_fraction = 0.25  # fraction of population considered as elite

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
        last_best_fitness = best_fitness

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                    + self.social_weight * r2 * (best_individual - population[i])
                )

                # Quantum behavior
                if np.random.rand() < self.quantum_weight:
                    quantum_step = np.random.normal(0, 1, self.dim)
                    population[i] = best_individual + 0.5 * quantum_step
                else:
                    population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                trial_fitness = evaluate(population[i])
                eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    personal_best_positions[i] = population[i]
                    personal_best_scores[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = population[i]
                        best_fitness = trial_fitness

                if eval_count >= self.budget:
                    break

            # Adaptive behavior
            if best_fitness < last_best_fitness * (1 - self.adaptive_threshold):
                self.quantum_weight *= 0.9
                last_best_fitness = best_fitness
            else:
                self.quantum_weight *= 1.1

            if eval_count < self.budget:
                elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
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
