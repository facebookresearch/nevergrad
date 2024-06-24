import numpy as np


class QuantumIterativeRefinementOptimizer:
    def __init__(self, budget, dim=5, pop_size=20, elite_rate=0.2, refinement_rate=0.95, quantum_prob=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.elite_count = int(pop_size * elite_rate)
        self.refinement_rate = refinement_rate
        self.quantum_prob = quantum_prob
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
                    self.best_solution = self.population[i]

    def refine_population(self):
        sorted_indices = np.argsort(self.fitnesses)
        elite_indices = sorted_indices[: self.elite_count]
        non_elite_indices = sorted_indices[self.elite_count :]

        # Refinement and reproduction from elites
        for idx in non_elite_indices:
            if np.random.rand() < self.quantum_prob:
                # Quantum jump inspired by best solution
                self.population[idx] = (
                    self.best_solution + np.random.normal(0, 0.1, self.dim) * self.refinement_rate
                )
            else:
                # Crossover and mutation
                parent1 = self.population[np.random.choice(elite_indices)]
                parent2 = self.population[np.random.choice(elite_indices)]
                crossover_point = np.random.randint(self.dim)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                mutation = np.random.normal(0, 0.1, self.dim) * (1 - self.refinement_rate)
                self.population[idx] = child + mutation

            # Ensure boundaries are respected
            self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.initialize()
        evaluations = self.pop_size

        while evaluations < self.budget:
            self.evaluate_fitness(func)
            self.refine_population()
            evaluations += self.pop_size

        return self.best_fitness, self.best_solution
