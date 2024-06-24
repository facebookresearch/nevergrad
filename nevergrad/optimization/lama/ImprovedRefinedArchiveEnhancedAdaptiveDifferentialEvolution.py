import numpy as np
from sklearn.cluster import KMeans


class ImprovedRefinedArchiveEnhancedAdaptiveDifferentialEvolution:
    def __init__(self, budget, population_size=20, mutation_factor=0.7, crossover_rate=0.9, cluster_size=5):
        self.budget = budget
        self.dim = 5
        self.bounds = [-5.0, 5.0]
        self.population_size = population_size
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.cluster_size = cluster_size
        self.epsilon = 1e-8
        self.archive = []

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
            kmeans = KMeans(n_clusters=self.cluster_size, random_state=0).fit(population)
            cluster_centers = kmeans.cluster_centers_
            for i in range(len(population)):
                if np.linalg.norm(population[i] - cluster_centers[kmeans.labels_[i]]) < 1e-1:
                    population[i] = random_vector()
                    fitness[i] = func(population[i])

        def select_parents(population, fitness):
            fitness = np.array(fitness)
            fitness = fitness - np.min(fitness) + 1e-8
            probabilities = 1 / fitness
            probabilities /= probabilities.sum()
            parents_idx = np.random.choice(np.arange(len(population)), size=3, p=probabilities, replace=False)
            return population[parents_idx[0]], population[parents_idx[1]], population[parents_idx[2]]

        def adaptive_params(success_rate):
            if success_rate > 0.2:
                new_mutation_factor = self.mutation_factor * 1.1
                new_crossover_rate = self.crossover_rate * 1.05
            else:
                new_mutation_factor = self.mutation_factor * 0.9
                new_crossover_rate = self.crossover_rate * 0.95
            return np.clip(new_mutation_factor, 0.4, 1.0), np.clip(new_crossover_rate, 0.5, 1.0)

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
            perturbation = np.random.randn(self.dim) * self.epsilon
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
        success_count_history = []

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
                        self.archive.append(self.x_opt)

            maintain_diversity(population, fitness)
            success_rate = success_count / self.population_size
            self.mutation_factor, self.crossover_rate = adaptive_params(success_rate)

            success_count_history.append(success_rate)
            if len(success_count_history) > 10:
                success_count_history.pop(0)

            avg_success_rate = np.mean(success_count_history)

            if avg_success_rate > 0.2:
                self.mutation_factor *= 1.1
                self.crossover_rate *= 1.05
            else:
                self.mutation_factor *= 0.9
                self.crossover_rate *= 0.95

            self.mutation_factor = np.clip(self.mutation_factor, 0.4, 1.0)
            self.crossover_rate = np.clip(self.crossover_rate, 0.5, 1.0)

            if len(self.archive) > 0:
                archive_selection = np.random.choice(len(self.archive))
                archive_mutant = np.clip(
                    self.archive[archive_selection] + self.mutation_factor * np.random.randn(self.dim),
                    self.bounds[0],
                    self.bounds[1],
                )
                archive_mutant = np.clip(archive_mutant, self.bounds[0], self.bounds[1])
                archive_fitness = func(archive_mutant)
                evaluations += 1
                if archive_fitness < self.f_opt:
                    self.f_opt = archive_fitness
                    self.x_opt = archive_mutant

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = ImprovedRefinedArchiveEnhancedAdaptiveDifferentialEvolution(budget=1000)
# best_value, best_solution = optimizer(some_black_box_function)
