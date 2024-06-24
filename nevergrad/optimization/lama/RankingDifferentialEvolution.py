import numpy as np


class RankingDifferentialEvolution:
    def __init__(self, budget=10000, population_size=30, f_weight=0.8, cr=0.9):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.f_weight = f_weight
        self.cr = cr

    def initialize_population(self):
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def generate_trial_vector(self, target_idx):
        candidates = np.random.choice(
            np.delete(np.arange(self.population_size), target_idx), size=3, replace=False
        )
        a, b, c = self.population[candidates]
        trial_vector = self.population[target_idx] + self.f_weight * (a - b)

        return np.clip(trial_vector, -5.0, 5.0)

    def update_population(self, func):
        for i in range(self.population_size):
            trial_vector = self.generate_trial_vector(i)
            crossover_mask = np.random.uniform(0, 1, self.dim) < self.cr
            new_vector = crossover_mask * trial_vector + (1 - crossover_mask) * self.population[i]

            new_value = func(new_vector)

            if new_value < func(self.population[i]):
                self.population[i] = new_vector

    def __call__(self, func):
        self.initialize_population()

        for _ in range(self.budget // self.population_size):
            self.update_population(func)

        best_idx = np.argmin([func(ind) for ind in self.population])
        best_solution = self.population[best_idx]

        return func(best_solution), best_solution
