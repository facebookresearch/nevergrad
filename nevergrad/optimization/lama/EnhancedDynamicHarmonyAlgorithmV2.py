import numpy as np


class EnhancedDynamicHarmonyAlgorithmV2:
    def __init__(
        self,
        budget=10000,
        population_size=20,
        dim=5,
        pa=0.25,
        beta=1.5,
        gamma=0.01,
        alpha=0.95,
        exploring_rate=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = dim
        self.pa = pa
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.exploring_rate = exploring_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.best_fitness = np.Inf
        self.best_solution = None

    def levy_flight(self):
        sigma1 = (
            np.math.gamma(1 + self.beta)
            * np.sin(np.pi * self.beta / 2)
            / (np.math.gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))
        ) ** (1 / self.beta)
        sigma2 = 1
        u = np.random.normal(0, sigma1, self.dim)
        v = np.random.normal(0, sigma2, self.dim)
        step = u / np.abs(v) ** (1 / self.beta)
        return step

    def calculate_fitness(self, func, solution):
        return func(solution)

    def update_population(self, func, iteration):
        harmony_pool = np.copy(self.population)
        new_harmony = np.zeros((self.population_size, self.dim))

        for i in range(self.population_size):
            if np.random.rand() < self.pa:
                # Perform Levy flight
                step = self.levy_flight()
                new_solution = self.population[i] + self.alpha * step
                new_solution_fitness = self.calculate_fitness(func, new_solution)
                if new_solution_fitness < self.best_fitness:
                    self.best_fitness = new_solution_fitness
                    self.best_solution = new_solution
                    harmony_pool[i] = new_solution

        # Diversify the population with new harmonies
        for i in range(self.population_size):
            j = np.random.randint(self.population_size)
            while j == i:
                j = np.random.randint(self.population_size)

            # Update current solution with harmony from another member
            new_harmony[i] = self.population[i] + self.gamma * (harmony_pool[j] - self.population[i])

            # Dynamic exploration rate adjustment
            exploring_rate = self.exploring_rate * np.exp(-iteration / self.budget)

            # Further exploration by random perturbation to improve diversity
            new_harmony[i] += np.random.normal(0, exploring_rate, self.dim)

            # Clamp new solutions within the search space bounds
            new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)

        self.population = new_harmony

    def __call__(self, func):
        for itr in range(1, self.budget + 1):
            self.update_population(func, itr)

        aocc = (
            1 - np.std(self.best_fitness) / np.mean(self.best_fitness)
            if np.mean(self.best_fitness) != 0
            else 0
        )
        return aocc, self.best_solution
