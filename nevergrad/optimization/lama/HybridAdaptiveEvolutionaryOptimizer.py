import numpy as np
from scipy.optimize import minimize


class HybridAdaptiveEvolutionaryOptimizer:
    def __init__(self, budget=10000, population_size=100):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.elite_fraction = 0.2
        self.local_search_probability = 0.8
        self.F = 0.8
        self.CR = 0.9
        self.memory_size = 10
        self.strategy_switch_threshold = 0.005
        self.rng = np.random.default_rng()
        self.num_strategies = 3

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        # Initialize population randomly within bounds
        population = self.rng.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        performance_memory = [best_fitness] * self.memory_size
        last_switch_eval_count = 0
        current_strategy = 0

        phase_one_budget = int(self.budget * 0.3)  # Increase exploration phase budget
        phase_two_budget = self.budget - phase_one_budget

        # Phase One: Hybrid Strategy Exploration
        while eval_count < phase_one_budget:
            new_population = np.copy(population)
            for i in range(self.population_size):
                if current_strategy == 0:
                    # Differential Evolution Strategy
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[self.rng.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                    cross_points = self.rng.random(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[self.rng.integers(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                elif current_strategy == 1:
                    # Particle Swarm Optimization Strategy
                    w = 0.5
                    c1 = 1.5
                    c2 = 1.5
                    r1 = self.rng.random(self.dim)
                    r2 = self.rng.random(self.dim)
                    velocity = (
                        w * self.rng.uniform(-1, 1, self.dim)
                        + c1 * r1 * (best_individual - population[i])
                        + c2 * r2 * (np.mean(population, axis=0) - population[i])
                    )
                    trial = np.clip(population[i] + velocity, self.bounds[0], self.bounds[1])
                else:
                    # Simulated Annealing Strategy
                    T = max(1e-10, (phase_one_budget - eval_count) / phase_one_budget)
                    neighbor = population[i] + self.rng.normal(0, 1, self.dim)
                    neighbor = np.clip(neighbor, self.bounds[0], self.bounds[1])
                    neighbor_fitness = evaluate(neighbor)
                    eval_count += 1
                    if neighbor_fitness < fitness[i] or self.rng.random() < np.exp(
                        (fitness[i] - neighbor_fitness) / T
                    ):
                        trial = neighbor
                    else:
                        trial = population[i]

                if current_strategy != 2:
                    trial_fitness = evaluate(trial)
                    eval_count += 1
                    if trial_fitness < fitness[i]:
                        new_population[i] = trial
                        fitness[i] = trial_fitness
                        if trial_fitness < best_fitness:
                            best_individual = trial
                            best_fitness = trial_fitness
                else:
                    if neighbor_fitness < fitness[i]:
                        new_population[i] = neighbor
                        fitness[i] = neighbor_fitness
                        if neighbor_fitness < best_fitness:
                            best_individual = neighbor
                            best_fitness = neighbor_fitness

                if eval_count >= phase_one_budget:
                    break

            population = new_population

            # Perform local search on elite individuals
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

            # Update performance memory and adapt strategy
            performance_memory.append(best_fitness)
            if len(performance_memory) > self.memory_size:
                performance_memory.pop(0)

            if eval_count - last_switch_eval_count >= self.memory_size:
                improvement = (performance_memory[0] - performance_memory[-1]) / max(
                    1e-10, performance_memory[0]
                )
                if improvement < self.strategy_switch_threshold:
                    current_strategy = (current_strategy + 1) % self.num_strategies
                    last_switch_eval_count = eval_count

        # Phase Two: Intensified Exploitation using Local Search
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
