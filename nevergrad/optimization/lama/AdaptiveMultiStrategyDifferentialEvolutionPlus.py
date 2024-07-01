import numpy as np


class AdaptiveMultiStrategyDifferentialEvolutionPlus:
    def __init__(self, budget, population_size=20, crossover_rate=0.9, mutation_factor=0.7):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_factor = mutation_factor
        self.base_lr = 0.1
        self.epsilon = 1e-8

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        def random_vector():
            return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

        def gradient_estimate(x, h=1e-5):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x1 = np.copy(x)
                x2 = np.copy(x)
                x1[i] += h
                x2[i] -= h
                grad[i] = (func(x1) - func(x2)) / (2 * h)
            return grad

        def maintain_diversity(population, fitness):
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    if np.linalg.norm(population[i] - population[j]) < 1e-3:
                        if fitness[i] > fitness[j]:
                            population[i] = random_vector()
                            fitness[i] = func(population[i])
                        else:
                            population[j] = random_vector()
                            fitness[j] = func(population[j])

        def select_parents(population, fitness):
            fitness = np.array(fitness)
            fitness = fitness - np.min(fitness) + 1e-8
            probabilities = 1 / fitness
            probabilities /= probabilities.sum()
            parents_idx = np.random.choice(np.arange(len(population)), size=3, p=probabilities, replace=False)
            return population[parents_idx[0]], population[parents_idx[1]], population[parents_idx[2]]

        def adaptive_lr(success_rate):
            if success_rate > 0.2:
                return self.base_lr * 1.1
            else:
                return self.base_lr * 0.9

        def levy_flight(Lambda):
            sigma = (
                np.math.gamma(1 + Lambda)
                * np.sin(np.pi * Lambda / 2)
                / (np.math.gamma((1 + Lambda) / 2) * Lambda * 2 ** ((Lambda - 1) / 2))
            ) ** (1 / Lambda)
            u = np.random.randn() * sigma
            v = np.random.randn()
            step = u / abs(v) ** (1 / Lambda)
            return 0.01 * step

        def dual_strategies(trial, grad):
            perturbation = np.random.randn(self.dim) * self.base_lr
            levy_step = levy_flight(1.5) * np.random.randn(self.dim)
            strategy_1 = trial - self.epsilon * grad + perturbation
            strategy_2 = trial + levy_step

            return strategy_1, strategy_2

        population = [random_vector() for _ in range(self.population_size)]
        fitness = [func(ind) for ind in population]

        for ind, fit in zip(population, fitness):
            if fit < self.f_opt:
                self.f_opt = fit
                self.x_opt = ind

        evaluations = len(population)
        success_rate = 0

        while evaluations < self.budget:
            success_count = 0

            for j in range(self.population_size):
                if evaluations >= self.budget:
                    break

                target = population[j]
                a, b, c = select_parents(population, fitness)
                mutant = np.clip(a + self.mutation_factor * (b - c), self.bounds[0], self.bounds[1])

                trial = np.copy(target)
                for k in range(self.dim):
                    if np.random.rand() < self.crossover_rate:
                        trial[k] = mutant[k]

                grad = gradient_estimate(trial)
                strategy_1, strategy_2 = dual_strategies(trial, grad)
                strategy_1 = np.clip(strategy_1, self.bounds[0], self.bounds[1])
                strategy_2 = np.clip(strategy_2, self.bounds[0], self.bounds[1])
                new_f1 = func(strategy_1)
                new_f2 = func(strategy_2)
                evaluations += 2

                if new_f1 < fitness[j] or new_f2 < fitness[j]:
                    if new_f1 < new_f2:
                        population[j] = strategy_1
                        fitness[j] = new_f1
                    else:
                        population[j] = strategy_2
                        fitness[j] = new_f2
                    success_count += 1

                if min(new_f1, new_f2) < self.f_opt:
                    self.f_opt = min(new_f1, new_f2)
                    self.x_opt = strategy_1 if new_f1 < new_f2 else strategy_2

            maintain_diversity(population, fitness)

            success_rate = success_count / self.population_size
            self.base_lr = adaptive_lr(success_rate)
            self.base_lr = np.clip(self.base_lr, 1e-4, 1.0)

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = AdaptiveMultiStrategyDifferentialEvolutionPlus(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
