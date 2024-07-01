import numpy as np
from sklearn.cluster import KMeans


class RefinedHybridDynamicClusterOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def levy_flight(self, dim, beta=1.5, alpha=0.01):
        sigma_u = (
            np.math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma_u, dim)
        v = np.random.normal(0, 1, dim)
        step = u / np.abs(v) ** (1 / beta)
        return alpha * step

    def adaptive_parameters(self, evaluations, max_evaluations, start_param, end_param):
        progress = evaluations / max_evaluations
        return start_param + (end_param - start_param) * progress

    def __call__(self, func):
        population_size = 100
        elite_size = 5
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        velocity = np.random.uniform(-0.1, 0.1, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        personal_best_positions = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best_position = population[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        self.f_opt = global_best_fitness
        self.x_opt = global_best_position

        while evaluations < self.budget:
            inertia_weight = self.adaptive_parameters(evaluations, self.budget, 0.9, 0.4)
            cognitive_coefficient = self.adaptive_parameters(evaluations, self.budget, 2.0, 1.0)
            social_coefficient = self.adaptive_parameters(evaluations, self.budget, 2.0, 1.0)
            differential_weight = self.adaptive_parameters(evaluations, self.budget, 0.8, 0.2)
            crossover_rate = self.adaptive_parameters(evaluations, self.budget, 0.9, 0.3)
            quantum_factor = self.adaptive_parameters(evaluations, self.budget, 0.5, 0.1)
            levy_factor = self.adaptive_parameters(evaluations, self.budget, 0.9, 0.4)
            cluster_count = int(self.adaptive_parameters(evaluations, self.budget, 2, 10))

            elite_indices = np.argsort(fitness)[:elite_size]
            elites = population[elite_indices]

            for i in range(population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                inertia = inertia_weight * velocity[i]
                cognitive = cognitive_coefficient * r1 * (personal_best_positions[i] - population[i])
                social = social_coefficient * r2 * (global_best_position - population[i])
                velocity[i] = inertia + cognitive + social
                new_position = np.clip(population[i] + velocity[i], self.lb, self.ub)
                new_fitness = func(new_position)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_position
                    fitness[i] = new_fitness

                    if new_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = new_position
                        personal_best_fitness[i] = new_fitness

                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_position

                indices = list(range(population_size))
                indices.remove(i)

                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = np.clip(a + differential_weight * (b - c), self.lb, self.ub)

                crossover_mask = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < personal_best_fitness[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_fitness[i] = trial_fitness

                        if trial_fitness < self.f_opt:
                            self.f_opt = trial_fitness
                            self.x_opt = trial_vector

            quantum_particles = population + quantum_factor * np.random.uniform(
                -1, 1, (population_size, self.dim)
            )
            quantum_particles = np.clip(quantum_particles, self.lb, self.ub)
            quantum_fitness = np.array([func(ind) for ind in quantum_particles])
            evaluations += population_size

            for i in range(population_size):
                if quantum_fitness[i] < fitness[i]:
                    population[i] = quantum_particles[i]
                    fitness[i] = quantum_fitness[i]

                    if quantum_fitness[i] < personal_best_fitness[i]:
                        personal_best_positions[i] = quantum_particles[i]
                        personal_best_fitness[i] = quantum_fitness[i]

                        if quantum_fitness[i] < self.f_opt:
                            self.f_opt = quantum_fitness[i]
                            self.x_opt = quantum_particles[i]

            kmeans = KMeans(n_clusters=cluster_count)
            clusters = kmeans.fit_predict(population)
            cluster_centers = kmeans.cluster_centers_

            for cluster_center in cluster_centers:
                levy_step = levy_factor * self.levy_flight(self.dim)
                candidate = np.clip(cluster_center + levy_step, self.lb, self.ub)
                candidate_fitness = func(candidate)
                evaluations += 1

                if candidate_fitness < self.f_opt:
                    self.f_opt = candidate_fitness
                    self.x_opt = candidate

            worst_indices = np.argsort(fitness)[-elite_size:]
            for idx in worst_indices:
                if evaluations < self.budget:
                    population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[idx] = func(population[idx])
                    evaluations += 1

                    if fitness[idx] < personal_best_fitness[idx]:
                        personal_best_positions[idx] = population[idx]
                        personal_best_fitness[idx] = fitness[idx]

                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = population[idx]

        return self.f_opt, self.x_opt
