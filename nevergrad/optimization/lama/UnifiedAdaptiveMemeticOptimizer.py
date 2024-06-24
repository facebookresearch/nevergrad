import numpy as np
from scipy.optimize import minimize


class UnifiedAdaptiveMemeticOptimizer:
    def __init__(self, budget=10000, population_size=200):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.elite_fraction = 0.1
        self.crossover_prob = 0.9
        self.mutation_prob = 0.2
        self.swarm_inertia = 0.6
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.strategy_switch_threshold = 0.005
        self.rng = np.random.default_rng()
        self.num_strategies = 4
        self.tol = 1e-6
        self.max_iter = 50
        self.mutation_factor = 0.8
        self.performance_memory = []
        self.memory_size = 30
        self.archive_size = 50
        self.learning_rate = 0.1
        self.min_local_search_iters = 10

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        population = self.rng.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        self.performance_memory = [best_fitness] * self.memory_size
        last_switch_eval_count = 0
        current_strategy = 0

        velocities = np.zeros((self.population_size, self.dim))
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        archive = np.copy(population[: self.archive_size])

        while eval_count < self.budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if current_strategy == 0:
                    # Genetic Algorithm
                    parent1, parent2 = population[self.rng.choice(self.population_size, 2, replace=False)]
                    cross_points = self.rng.random(self.dim) < self.crossover_prob
                    if not np.any(cross_points):
                        cross_points[self.rng.integers(0, self.dim)] = True
                    child = np.where(cross_points, parent1, parent2)
                    mutation = self.rng.uniform(self.bounds[0], self.bounds[1], self.dim)
                    mutate = self.rng.random(self.dim) < self.mutation_prob
                    trial = np.where(mutate, mutation, child)
                elif current_strategy == 1:
                    # Particle Swarm Optimization
                    r1 = self.rng.random(self.dim)
                    r2 = self.rng.random(self.dim)
                    velocities[i] = (
                        self.swarm_inertia * velocities[i]
                        + self.cognitive_coeff * r1 * (personal_best[i] - population[i])
                        + self.social_coeff * r2 * (best_individual - population[i])
                    )
                    trial = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])
                elif current_strategy == 2:
                    # Differential Evolution
                    indices = self.rng.choice(self.population_size, 3, replace=False)
                    x0, x1, x2 = population[indices]
                    mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), self.bounds[0], self.bounds[1])
                    cross_points = self.rng.random(self.dim) < self.crossover_prob
                    if not np.any(cross_points):
                        cross_points[self.rng.integers(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                else:
                    # Memetic Algorithm with Local Search
                    elite_index = np.argmin(fitness)
                    trial = population[elite_index] + self.learning_rate * (self.rng.random(self.dim) - 0.5)
                    trial = np.clip(trial, self.bounds[0], self.bounds[1])
                    trial_fitness = evaluate(trial)
                    eval_count += 1
                    if trial_fitness < fitness[i]:
                        new_population[i] = trial
                        fitness[i] = trial_fitness

                if current_strategy != 3:
                    trial_fitness = evaluate(trial)
                    eval_count += 1
                    if trial_fitness < fitness[i]:
                        new_population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < personal_best_fitness[i]:
                            personal_best[i] = trial
                            personal_best_fitness[i] = trial_fitness
                            if trial_fitness < best_fitness:
                                best_individual = trial
                                best_fitness = trial_fitness

            population = new_population

            # Memory-based archive learning
            if best_fitness not in fitness:
                if len(archive) < self.archive_size:
                    archive = np.vstack([archive, best_individual])
                else:
                    worst_index = np.argmax(np.array([evaluate(ind) for ind in archive]))
                    if best_fitness < archive[worst_index]:
                        archive[worst_index] = best_individual

            elite_count = max(1, int(self.population_size * self.elite_fraction))
            elite_indices = np.argsort(fitness)[:elite_count]
            for idx in elite_indices:
                res = self.local_search(func, population[idx])
                if res is not None:
                    eval_count += 1
                    if res[1] < fitness[idx]:
                        population[idx] = res[0]
                        fitness[idx] = res[1]
                        if res[1] < best_fitness:
                            best_individual = res[0]
                            best_fitness = res[1]

            self.performance_memory.append(best_fitness)
            if len(self.performance_memory) > self.memory_size:
                self.performance_memory.pop(0)

            if eval_count - last_switch_eval_count >= self.memory_size:
                improvement = (self.performance_memory[0] - self.performance_memory[-1]) / max(
                    1e-10, self.performance_memory[0]
                )
                if improvement < self.strategy_switch_threshold:
                    current_strategy = (current_strategy + 1) % self.num_strategies
                    last_switch_eval_count = eval_count

        self.f_opt = best_fitness
        self.x_opt = best_individual
        return self.f_opt, self.x_opt

    def local_search(self, func, x_start):
        res = minimize(
            func,
            x_start,
            method="L-BFGS-B",
            bounds=[self.bounds] * self.dim,
            tol=self.tol,
            options={"maxiter": self.min_local_search_iters},
        )
        if res.success:
            return res.x, res.fun
        return None
