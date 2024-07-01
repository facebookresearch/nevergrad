import numpy as np


class EnhancedQuantumDynamicBalanceOptimizer:
    def __init__(
        self,
        budget,
        dim=5,
        learning_rate=0.1,
        momentum=0.95,
        quantum_prob=0.15,
        elite_rate=0.1,
        noise_intensity=0.05,
        perturbation_scale=0.05,
        balance_factor=0.75,
        adaptive_lr=True,
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
        self.balance_factor = balance_factor  # Adjusted for better quantum/classical balance
        self.adaptive_lr = adaptive_lr  # New feature for adaptive learning rates

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
                # Quantum jump influenced by dynamic balance factor
                quantum_jump = np.random.normal(
                    0.0,
                    self.perturbation_scale
                    * (1 - self.balance_factor)
                    * np.linalg.norm(global_best - self.population[i]),
                    self.dim,
                )
                self.population[i] += quantum_jump
            else:
                # Classic momentum and adaptive learning rate
                if self.adaptive_lr:
                    lr = self.learning_rate / (
                        1 + 0.01 * np.sqrt(np.sum((global_best - self.population[i]) ** 2))
                    )
                else:
                    lr = self.learning_rate
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
