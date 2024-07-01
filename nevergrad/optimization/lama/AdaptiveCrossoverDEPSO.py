import numpy as np
from scipy.optimize import minimize


class AdaptiveCrossoverDEPSO:
    def __init__(self, budget=10000, population_size=200):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.elite_fraction = 0.2
        self.local_search_probability = 0.95
        self.crossover_prob = 0.9
        self.mutation_prob = 0.1
        self.swarm_inertia = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.strategy_switch_threshold = 0.01
        self.rng = np.random.default_rng()
        self.num_strategies = 2
        self.tol = 1e-6
        self.max_iter = 50
        self.mutation_factor = 0.8
        self.adaptive_crossover_prob = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.performance_memory = []
        self.memory_size = 30
        self.archive_size = 50  # Archive size for memory-based learning

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

        phase_one_budget = int(self.budget * 0.6)
        phase_two_budget = self.budget - phase_one_budget

        while eval_count < phase_one_budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if current_strategy == 0:
                    parent1, parent2 = population[self.rng.choice(self.population_size, 2, replace=False)]
                    cross_points = (
                        self.rng.random(self.dim)
                        < self.adaptive_crossover_prob[i % len(self.adaptive_crossover_prob)]
                    )
                    if not np.any(cross_points):
                        cross_points[self.rng.integers(0, self.dim)] = True
                    child = np.where(cross_points, parent1, parent2)
                    mutation = self.rng.uniform(self.bounds[0], self.bounds[1], self.dim)
                    mutate = self.rng.random(self.dim) < self.mutation_prob
                    trial = np.where(mutate, mutation, child)
                else:
                    r1 = self.rng.random(self.dim)
                    r2 = self.rng.random(self.dim)
                    velocities[i] = (
                        self.swarm_inertia * velocities[i]
                        + self.cognitive_coeff * r1 * (personal_best[i] - population[i])
                        + self.social_coeff * r2 * (best_individual - population[i])
                    )
                    trial = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

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
            archive_fitness = np.array([evaluate(ind) for ind in archive])
            eval_count += len(archive)
            if best_fitness not in archive_fitness:
                worst_index = np.argmax(archive_fitness)
                if best_fitness < archive_fitness[worst_index]:
                    archive[worst_index] = best_individual

            elite_count = max(1, int(self.population_size * self.elite_fraction))
            elite_indices = np.argsort(fitness)[:elite_count]
            for idx in elite_indices:
                if self.rng.random() < self.local_search_probability:
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

            if eval_count >= self.budget:
                break

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
            options={"maxiter": self.max_iter},
        )
        if res.success:
            return res.x, res.fun
        return None
