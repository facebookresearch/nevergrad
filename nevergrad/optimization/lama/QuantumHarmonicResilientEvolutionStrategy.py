import numpy as np


class QuantumHarmonicResilientEvolutionStrategy:
    def __init__(
        self,
        budget,
        dim=5,
        pop_size=150,
        elite_rate=0.20,
        resonance_intensity=0.12,
        mutation_intensity=0.03,
        harmonic_depth=0.25,
        feedback_factor=0.1,
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.elite_count = int(pop_size * elite_rate)
        self.resonance_intensity = resonance_intensity
        self.mutation_intensity = mutation_intensity
        self.harmonic_depth = harmonic_depth
        self.feedback_factor = feedback_factor
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
        # Sort population by fitness and select elites
        sorted_indices = np.argsort(self.fitnesses)
        elite_indices = sorted_indices[: self.elite_count]
        non_elite_indices = sorted_indices[self.elite_count :]

        # Generate new solutions based on elites with harmonic fluctuations and quantum resonance
        for idx in non_elite_indices:
            elite_sample = self.population[np.random.choice(elite_indices)]
            harmonic_influence = self.harmonic_depth * np.sin(np.random.uniform(0, 2 * np.pi, self.dim))
            quantum_resonance = self.resonance_intensity * (np.random.uniform(-1, 1, self.dim) ** 3)
            normal_disturbance = np.random.normal(0, self.mutation_intensity, self.dim)

            # Feedback mechanism: adapt to stagnation
            if self.best_fitness >= self.prev_best_fitness:
                feedback_adjustment = self.feedback_factor * np.random.uniform(-1, 1, self.dim)
                self.harmonic_depth *= 0.95  # Dampen harmonic depth to refocus search
            else:
                feedback_adjustment = 0

            # Combine influences
            self.population[idx] = (
                elite_sample
                + harmonic_influence
                + quantum_resonance
                + normal_disturbance
                + feedback_adjustment
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
