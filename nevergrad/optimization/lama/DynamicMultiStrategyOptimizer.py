import numpy as np
from scipy.optimize import minimize


class DynamicMultiStrategyOptimizer:
    def __init__(self, budget=10000, population_size=100):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.elite_fraction = 0.2
        self.local_search_probability = 0.8
        self.F = 0.8
        self.CR = 0.9
        self.memory_size = 50
        self.strategy_switch_threshold = 0.1

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        performance_memory = [best_fitness] * self.memory_size

        last_switch_eval_count = 0
        use_de_strategy = True

        while eval_count < self.budget:
            new_population = []
            for i in range(self.population_size):
                if use_de_strategy:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                    cross_points = np.random.rand(self.dim) < self.CR
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, population[i])
                else:
                    w = 0.5
                    c1 = 1.5
                    c2 = 1.5
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    velocity = (
                        w * np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
                        + c1 * r1 * (best_individual - population[i])
                        + c2 * r2 * (np.mean(population, axis=0) - population[i])
                    )
                    trial = np.clip(population[i] + velocity, self.bounds[0], self.bounds[1])

                trial_fitness = evaluate(trial)
                eval_count += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_individual = trial
                        best_fitness = trial_fitness
                else:
                    new_population.append(population[i])

                if eval_count >= self.budget:
                    break

            population = np.array(new_population)

            elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
            for idx in elite_indices:
                if np.random.rand() < self.local_search_probability:
                    res = self.local_search(func, population[idx])
                    if res is not None:
                        eval_count += 1
                        if res[1] < fitness[idx]:
                            population[idx] = res[0]
                            fitness[idx] = res[1]
                            if res[1] < best_fitness:
                                best_individual = res[0]
                                best_fitness = res[1]

            performance_memory.append(best_fitness)
            if len(performance_memory) > self.memory_size:
                performance_memory.pop(0)

            if eval_count - last_switch_eval_count >= self.memory_size:
                improvement = (performance_memory[0] - performance_memory[-1]) / max(
                    1e-10, performance_memory[0]
                )
                if improvement < self.strategy_switch_threshold:
                    use_de_strategy = not use_de_strategy
                    last_switch_eval_count = eval_count

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


# Example usage
# optimizer = DynamicMultiStrategyOptimizer(budget=10000)
# best_fitness, best_solution = optimizer(some_black_box_function)
