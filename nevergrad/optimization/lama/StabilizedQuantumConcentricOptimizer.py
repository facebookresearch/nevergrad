import numpy as np


class StabilizedQuantumConcentricOptimizer:
    def __init__(
        self,
        budget,
        dim=5,
        pop_size=100,
        elite_rate=0.2,
        initial_mutation_scale=0.5,
        mutation_decay_factor=0.95,
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.elite_count = int(pop_size * elite_rate)
        self.initial_mutation_scale = initial_mutation_scale
        self.mutation_decay_factor = mutation_decay_factor
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitnesses = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.mutation_scale = self.initial_mutation_scale

    def evaluate_fitness(self, func):
        for i in range(self.pop_size):
            fitness = func(self.population[i])
            if fitness < self.fitnesses[i]:
                self.fitnesses[i] = fitness
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = np.copy(self.population[i])

    def update_population(self):
        # Sort population by fitness and select elites
        sorted_indices = np.argsort(self.fitnesses)
        elite_indices = sorted_indices[: self.elite_count]
        non_elite_indices = sorted_indices[self.elite_count :]

        # Generate new solutions based on elites
        for idx in non_elite_indices:
            elite_sample = self.population[np.random.choice(elite_indices)]
            deviation = np.random.normal(
                0, max(self.mutation_scale, 0.001), self.dim
            )  # Ensuring non-negative scale
            self.population[idx] = elite_sample + deviation
            self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

        # Reduce mutation scale
        self.mutation_scale *= self.mutation_decay_factor

    def __call__(self, func):
        self.initialize()
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_fitness(func)
            self.update_population()
            evaluations += self.pop_size

        return self.best_fitness, self.best_solution
