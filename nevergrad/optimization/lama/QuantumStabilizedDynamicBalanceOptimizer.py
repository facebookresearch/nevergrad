import numpy as np


class QuantumStabilizedDynamicBalanceOptimizer:
    def __init__(
        self,
        budget,
        dim=5,
        learning_rate=0.05,
        momentum=0.9,
        quantum_prob=0.1,
        elite_rate=0.2,
        noise_intensity=0.1,
        perturbation_scale=0.1,
        stability_factor=0.8,
        decay_rate=0.005,
    ):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.quantum_prob = quantum_prob
        self.elite_rate = elite_rate
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.noise_intensity = noise_intensity
        self.perturbation_scale = perturbation_scale
        self.stability_factor = stability_factor  # Increased stability in the updating mechanism
        self.decay_rate = decay_rate  # Decay rate for adaptive learning rate

    def initialize(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (int(self.budget * self.elite_rate), self.dim)
        )
        self.velocities = np.zeros((int(self.budget * self.elite_rate), self.dim))
        self.fitnesses = np.full(int(self.budget * self.elite_rate), np.inf)

    def evaluate_population(self, func):
        for i in range(len(self.population)):
            fitness = func(self.population[i])
            if fitness < self.fitnesses[i]:
                self.fitnesses[i] = fitness

    def update_particles(self):
        best_idx = np.argmin(self.fitnesses)
        global_best = self.population[best_idx]

        for i in range(len(self.population)):
            if np.random.rand() < self.quantum_prob:
                quantum_jump = np.random.normal(
                    0.0,
                    self.perturbation_scale * np.exp(-self.stability_factor * self.decay_rate * i),
                    self.dim,
                )
                self.population[i] += quantum_jump
            else:
                lr = self.learning_rate * np.exp(-self.decay_rate * i)  # Exponentially decaying learning rate
                noise = np.random.normal(0, self.noise_intensity, self.dim)
                self.velocities[i] = (
                    self.momentum * self.velocities[i] + lr * (global_best - self.population[i]) + noise
                )
                future_position = self.population[i] + self.velocities[i]
                future_position = np.clip(future_position, self.lower_bound, self.upper_bound)
                self.population[i] = future_position

    def __call__(self, func):
        self.initialize()
        total_evaluations = len(self.population)
        while total_evaluations < self.budget:
            self.evaluate_population(func)
            self.update_particles()
            total_evaluations += len(self.population)

        best_idx = np.argmin(self.fitnesses)
        return self.fitnesses[best_idx], self.population[best_idx]
