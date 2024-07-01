import numpy as np


class QuantumHarmonicAdaptiveOptimizer:
    def __init__(
        self,
        budget,
        dim=5,
        pop_size=100,
        elite_rate=0.1,
        resonance_factor=0.05,
        mutation_scale=0.05,
        harmonic_frequency=0.1,
        feedback_intensity=0.1,
        damping_factor=0.98,
        mutation_decay=0.98,
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.elite_count = int(pop_size * elite_rate)
        self.resonance_factor = resonance_factor
        self.mutation_scale = mutation_scale
        self.harmonic_frequency = harmonic_frequency
        self.feedback_intensity = feedback_intensity
        self.damping_factor = damping_factor
        self.mutation_decay = mutation_decay
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
        # Sort population by fitness and perform selective reproduction
        sorted_indices = np.argsort(self.fitnesses)
        elite_indices = sorted_indices[: self.elite_count]
        non_elite_indices = sorted_indices[self.elite_count :]

        # Generate new solutions based on elites with quantum-inspired variations
        for idx in non_elite_indices:
            elite_sample = self.population[np.random.choice(elite_indices)]
            harmonic_influence = self.harmonic_frequency * np.sin(np.random.uniform(0, 2 * np.pi, self.dim))
            quantum_resonance = self.resonance_factor * (np.random.uniform(-1, 1, self.dim) ** 3)
            mutation_effect = np.random.normal(0, self.mutation_scale, self.dim)

            self.population[idx] = elite_sample + harmonic_influence + quantum_resonance + mutation_effect
            self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

            # Decay mutation parameters to stabilize convergence over time
            self.mutation_scale *= self.mutation_decay

    def __call__(self, func):
        self.initialize()
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_fitness(func)
            self.update_population()
            evaluations += self.pop_size

        return self.best_fitness, self.best_solution
