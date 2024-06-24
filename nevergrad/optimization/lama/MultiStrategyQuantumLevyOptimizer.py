import numpy as np
from scipy.optimize import minimize


class MultiStrategyQuantumLevyOptimizer:
    def __init__(self, budget=10000, population_size=100):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.inertia_weight = 0.7
        self.cognitive_weight = 1.4
        self.social_weight = 1.4
        self.quantum_weight = 0.35
        self.elite_fraction = 0.2
        self.memory_size = 20
        self.local_search_probability = 0.6
        self.stagnation_threshold = 5
        self.adaptive_factor = 1.1
        self.no_improvement_count = 0
        self.annealing_factor = 0.95
        self.strategy_probabilities = [1 / 4, 1 / 4, 1 / 4, 1 / 4]
        self.strategy_rewards = [0, 0, 0, 0]
        self.strategy_uses = [0, 0, 0, 0]

    def levy_flight(self, size, beta=1.5):
        sigma_u = (
            np.math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=size)
        v = np.random.normal(0, 1, size=size)
        step = u / abs(v) ** (1 / beta)
        return 0.01 * step

    def select_strategy(self):
        return np.random.choice([0, 1, 2, 3], p=self.strategy_probabilities)

    def update_strategy_probabilities(self):
        total_rewards = sum(self.strategy_rewards)
        if total_rewards > 0:
            self.strategy_probabilities = [r / total_rewards for r in self.strategy_rewards]
        else:
            self.strategy_probabilities = [1 / 4, 1 / 4, 1 / 4, 1 / 4]

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
        performance_memory = [best_fitness] * self.memory_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                strategy = self.select_strategy()
                if strategy == 0:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = (
                        self.inertia_weight * velocities[i]
                        + self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                        + self.social_weight * r2 * (best_individual - population[i])
                    )
                    population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])
                elif strategy == 1:
                    if np.random.rand() < self.quantum_weight:
                        levy_step = self.levy_flight(self.dim)
                        step_size = np.linalg.norm(velocities[i])
                        population[i] = best_individual + step_size * levy_step
                    else:
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        velocities[i] = (
                            self.inertia_weight * velocities[i]
                            + self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                            + self.social_weight * r2 * (best_individual - population[i])
                        )
                        population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])
                elif strategy == 2:
                    if np.random.rand() < self.local_search_probability:
                        new_population = self.local_search(func, population[i])
                        if new_population is not None:
                            population[i], fitness[i] = new_population
                            eval_count += 1
                    else:
                        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                        velocities[i] = (
                            self.inertia_weight * velocities[i]
                            + self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                            + self.social_weight * r2 * (best_individual - population[i])
                        )
                        population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])
                elif strategy == 3:
                    elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
                    for idx in elite_indices:
                        if np.random.rand() < self.local_search_probability:
                            res = self.local_search(func, population[idx])
                            if res is not None:
                                eval_count += 1
                                if res[1] < fitness[idx]:
                                    population[idx] = res[0]
                                    fitness[idx] = res[1]
                                    personal_best_positions[idx] = res[0]
                                    personal_best_scores[idx] = res[1]
                                    if res[1] < best_fitness:
                                        best_individual = res[0]
                                        best_fitness = res[1]
                                        self.no_improvement_count = 0

                trial_fitness = evaluate(population[i])
                eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    personal_best_positions[i] = population[i]
                    personal_best_scores[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = population[i]
                        best_fitness = trial_fitness
                        self.no_improvement_count = 0
                    else:
                        self.no_improvement_count += 1
                else:
                    self.no_improvement_count += 1

                self.strategy_rewards[strategy] += best_fitness - trial_fitness
                self.strategy_uses[strategy] += 1

                if eval_count >= self.budget:
                    break

            performance_memory.append(best_fitness)
            if len(performance_memory) > self.memory_size:
                performance_memory.pop(0)

            mean_recent_performance = np.mean(performance_memory)
            if best_fitness > mean_recent_performance * 1.05:
                self.adaptive_factor *= 0.9
                self.quantum_weight = min(1.0, self.quantum_weight * self.adaptive_factor)
            else:
                self.adaptive_factor *= 1.1
                self.quantum_weight = max(0.0, self.quantum_weight * self.adaptive_factor)

            if self.no_improvement_count >= self.stagnation_threshold:
                elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
                for idx in elite_indices:
                    if np.random.rand() < self.local_search_probability:
                        res = self.local_search(func, population[idx])
                        if res is not None:
                            eval_count += 1
                            if res[1] < fitness[idx]:
                                population[idx] = res[0]
                                fitness[idx] = res[1]
                                personal_best_positions[idx] = res[0]
                                personal_best_scores[idx] = res[1]
                                if res[1] < best_fitness:
                                    best_individual = res[0]
                                    best_fitness = res[1]
                                    self.no_improvement_count = 0

                    if eval_count >= self.budget:
                        break

                self.no_improvement_count = 0

            self.inertia_weight *= self.annealing_factor

            self.update_strategy_probabilities()

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
# optimizer = MultiStrategyQuantumLevyOptimizer(budget=10000)
# best_fitness, best_solution = optimizer(some_black_box_function)
