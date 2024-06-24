import numpy as np


class EnhancedDynamicHarmonySearchOptimizer:
    def __init__(
        self, budget=10000, population_size=20, dim=5, pa=0.1, hmcr=0.7, bw=0.01, exploring_rate=0.1
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = dim
        self.pa = pa
        self.hmcr = hmcr
        self.bw = bw
        self.exploring_rate = exploring_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.best_fitness = np.Inf
        self.best_solution = None

    def levy_flight(self):
        sigma1 = 1.0
        sigma2 = 1.0
        u = np.random.normal(0, sigma1, self.dim)
        v = np.random.normal(0, sigma2, self.dim)
        step = u / np.abs(v)
        return step

    def calculate_fitness(self, func, solution):
        return func(solution)

    def update_population(self, func, iteration):
        harmony_pool = np.copy(self.population)
        new_harmony = np.zeros((self.population_size, self.dim))

        for i in range(self.population_size):
            if np.random.rand() < self.pa:
                # Perform Harmony Memory Consideration Rate (HMCR)
                if np.random.rand() < self.hmcr:
                    j = np.random.randint(self.population_size)
                    new_solution = harmony_pool[j]
                else:
                    # Perform random selection from harmony memory
                    new_solution = harmony_pool[np.random.randint(self.population_size)]

                # Pitch Adjustment Rate (PAR)
                for k in range(self.dim):
                    if np.random.rand() < self.bw:
                        new_solution[k] = new_solution[k] + self.levy_flight()[k]

                new_solution_fitness = self.calculate_fitness(func, new_solution)
                if new_solution_fitness < self.best_fitness:
                    self.best_fitness = new_solution_fitness
                    self.best_solution = new_solution
                    harmony_pool[i] = new_solution

        # Diversify the population with new harmonies
        for i in range(self.population_size):
            if np.random.rand() < self.exploring_rate:
                new_harmony[i] = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                new_harmony[i] = harmony_pool[np.random.randint(self.population_size)]

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
