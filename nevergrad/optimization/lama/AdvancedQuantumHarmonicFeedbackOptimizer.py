import numpy as np


class AdvancedQuantumHarmonicFeedbackOptimizer:
    def __init__(
        self,
        budget,
        dim=5,
        pop_size=250,
        elite_rate=0.20,
        resonance_factor=0.08,
        mutation_scale=0.03,
        harmonic_frequency=0.25,
        feedback_intensity=0.15,
        damping_factor=0.95,
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
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.fitnesses = np.full(self.pop_size, np.inf)
        self.best_solution = None
        self.best_fitness = np.inf
        self.prev_best_fitness = np.inf

    def evaluate_fitness(self, func):
        for i in range(self.pop_size):
            fitness = func(self.population[i])
            if fitness < self.fitnesses[i]:
                self.fitnesses[i] = fitness
                if fitness < self.best_fitness:
                    self.prev_best_fitness = self.best_fitness
                    self.best_fitness = fitness
                    self.best_solution = np.copy(self.population[i])

    def update_population(self):
        # Sort population by fitness and perform a selective reproduction process
        sorted_indices = np.argsort(self.fitnesses)
        elite_indices = sorted_indices[: self.elite_count]
        non_elite_indices = sorted_indices[self.elite_count :]

        # Apply harmonic and quantum techniques with feedback and damping
        for idx in non_elite_indices:
            elite_sample = self.population[np.random.choice(elite_indices)]
            harmonic_influence = self.harmonic_frequency * np.sin(np.random.uniform(0, 2 * np.pi, self.dim))
            quantum_resonance = self.resonance_factor * (np.random.uniform(-1, 1, self.dim) ** 3)
            mutation_effect = np.random.normal(0, self.mutation_scale, self.dim)

            if self.best_fitness >= self.prev_best_fitness:
                feedback_adjustment = self.feedback_intensity * np.random.uniform(-1, 1, self.dim)
            else:
                feedback_adjustment = 0

            # Apply a damping factor to gradually reduce the scale of random perturbations
            self.population[idx] = (
                elite_sample
                + (harmonic_influence + quantum_resonance + mutation_effect + feedback_adjustment)
                * self.damping_factor
            )
            self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.initialize()
        evaluations = 0
        while evaluations < self.budget:
            self.evaluate_fitness(func)
            self.update_population()
            evaluations += self.pop_size

        return self.best_fitness, self.best_solution
