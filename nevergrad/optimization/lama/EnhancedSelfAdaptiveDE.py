import numpy as np


class EnhancedSelfAdaptiveDE:
    def __init__(self, budget=10000, population_size=30, c=0.1, p=0.05):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.c = c
        self.p = p

    def initialize_population(self):
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def generate_trial_vector(self, target_idx, scaling_factor):
        candidates = np.random.choice(
            np.delete(np.arange(self.population_size), target_idx), size=2, replace=False
        )
        a, b = self.population[candidates]
        mutant_vector = self.population[target_idx] + scaling_factor * (a - b)

        return np.clip(mutant_vector, -5.0, 5.0)

    def update_population(self, func):
        for i in range(self.population_size):
            scaling_factor = np.random.normal(self.c, self.p)
            trial_vector = self.generate_trial_vector(i, scaling_factor)
            new_value = func(trial_vector)

            if new_value < func(self.population[i]):
                self.population[i] = trial_vector

    def adapt_parameters(self, func):
        best_idx = np.argmin([func(ind) for ind in self.population])
        best_solution = self.population[best_idx]

        for i in range(self.population_size):
            scaling_factor = np.random.normal(self.c, self.p)
            trial_vector = self.generate_trial_vector(i, scaling_factor)
            trial_value = func(trial_vector)

            if trial_value < func(self.population[i]):
                self.c = (1 - self.p) * self.c + self.p * scaling_factor
                self.p = (1 - self.p) * self.p + self.p * int(trial_value < func(best_solution))

    def __call__(self, func):
        self.initialize_population()

        for _ in range(self.budget // self.population_size):
            self.update_population(func)
            self.adapt_parameters(func)

        best_idx = np.argmin([func(ind) for ind in self.population])
        best_solution = self.population[best_idx]

        return func(best_solution), best_solution
