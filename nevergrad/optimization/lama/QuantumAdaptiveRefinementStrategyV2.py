import numpy as np


class QuantumAdaptiveRefinementStrategyV2:
    def __init__(
        self,
        budget,
        dim=5,
        pop_size=30,
        elite_rate=0.15,
        mutation_scale=0.1,
        quantum_jump_scale=0.05,
        adaptation_factor=0.99,
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.elite_count = int(pop_size * elite_rate)
        self.mutation_scale = mutation_scale
        self.quantum_jump_scale = quantum_jump_scale
        self.adaptation_factor = adaptation_factor
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

    def refine_population(self):
        sorted_indices = np.argsort(self.fitnesses)
        elite_indices = sorted_indices[: self.elite_count]
        non_elite_indices = sorted_indices[self.elite_count :]

        # Adaptive mutation and quantum jump scale
        self.mutation_scale *= self.adaptation_factor
        self.quantum_jump_scale *= self.adaptation_factor

        # Refinement and reproduction from elites
        for idx in non_elite_indices:
            if np.random.rand() < self.adaptation_factor:  # Increasingly favor quantum jumps over time
                # Quantum jump inspired by best solution
                self.population[idx] = self.best_solution + np.random.normal(
                    0, self.quantum_jump_scale, self.dim
                )
            else:
                # Crossover and mutation
                parent1 = self.population[np.random.choice(elite_indices)]
                parent2 = self.population[np.random.choice(elite_indices)]
                crossover_point = np.random.randint(self.dim)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                mutation = np.random.normal(0, self.mutation_scale, self.dim)
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
