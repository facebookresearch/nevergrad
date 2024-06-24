import numpy as np


class AdaptiveQuantumResonanceOptimizer:
    def __init__(self, budget, dim=5, pop_size=100, learning_rate=0.1, elite_rate=0.1, resonance_factor=0.05):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.elite_count = int(pop_size * elite_rate)
        self.learning_rate = learning_rate
        self.resonance_factor = resonance_factor
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitnesses = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf

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

        # Generate new solutions based on elites with a resonance factor
        for idx in non_elite_indices:
            elite_sample = self.population[np.random.choice(elite_indices)]
            random_resonance = np.random.uniform(-self.resonance_factor, self.resonance_factor, self.dim)
            mutation = np.random.normal(0, self.resonance_factor, self.dim)
            self.population[idx] = elite_sample + mutation + self.learning_rate * random_resonance
            self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.initialize()
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_fitness(func)
            self.update_population()
            evaluations += self.pop_size

        return self.best_fitness, self.best_solution
