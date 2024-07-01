import numpy as np


class NovelEnhancedDiversifiedMetaHeuristicAlgorithmV2:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        num_iterations=100,
        mutation_rate=0.1,
        step_size=0.1,
        diversity_rate=0.2,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.step_size = step_size
        self.diversity_rate = diversity_rate

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def levy_flight(self):
        beta = 1.5
        sigma1 = (
            np.math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        sigma2 = 1
        u = np.random.normal(0, sigma1, self.dim)
        v = np.random.normal(0, sigma2, self.dim)
        levy = u / (np.abs(v) ** (1 / beta))
        return levy

    def adaptive_mutation_rate(self, success_counts, trial_counts):
        return self.mutation_rate * (1 - success_counts / (trial_counts + 1))

    def update_trial_counts(self, success_mask, trial_counts):
        trial_counts += ~success_mask
        trial_counts[success_mask] = 0
        return trial_counts

    def diversity_mutation(self, population):
        mask = np.random.rand(self.population_size, self.dim) < self.diversity_rate
        new_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        population = np.where(mask, new_population, population)
        return population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(sol) for sol in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        success_counts = np.zeros(self.population_size)
        trial_counts = np.zeros(self.population_size)

        for _ in range(self.budget // self.population_size):
            offspring_population = []
            for _ in range(self.population_size):
                new_solution = best_solution + self.step_size * self.levy_flight()
                offspring_population.append(new_solution)

            population = np.vstack((population, offspring_population))
            fitness = np.array([func(sol) for sol in population])
            sorted_indices = np.argsort(fitness)[: self.population_size]
            population = population[sorted_indices]
            fitness = np.array([func(sol) for sol in population])

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]
                success_counts += 1

            success_mask = fitness < best_fitness
            trial_counts = self.update_trial_counts(success_mask, trial_counts)
            mutation_rates = self.adaptive_mutation_rate(success_counts, trial_counts)

            population = self.diversity_mutation(population)
            self.step_size = np.clip(self.step_size * np.exp(np.mean(mutation_rates)), 0.01, 0.5)

        return best_fitness, best_solution
