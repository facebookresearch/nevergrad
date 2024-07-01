import numpy as np


class QuantumAdaptiveNesterovGradientEnhancer:
    def __init__(
        self,
        budget,
        dim=5,
        learning_rate=0.15,
        momentum=0.95,
        quantum_influence_rate=0.03,
        adaptive_lr_factor=0.98,
    ):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.quantum_influence_rate = quantum_influence_rate
        self.adaptive_lr_factor = adaptive_lr_factor
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize(self):
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
        self.velocity = np.zeros(self.dim)
        self.best_position = np.copy(self.position)
        self.best_fitness = np.inf

    def evaluate(self, func, position):
        return func(position)

    def update_position(self):
        # Predict future position using current velocity (Nesterov acceleration)
        future_position = self.position + self.momentum * self.velocity
        future_position = np.clip(future_position, self.lower_bound, self.upper_bound)

        # Update velocity with noise as a surrogate gradient and include Nesterov correction
        noise = np.random.normal(0, 1, self.dim) * self.learning_rate
        self.velocity = self.momentum * self.velocity - noise
        self.position += self.velocity
        self.position = np.clip(self.position, self.lower_bound, self.upper_bound)

        # Adaptive learning rate decay
        self.learning_rate *= self.adaptive_lr_factor

    def quantum_influence(self):
        if np.random.rand() < self.quantum_influence_rate:
            quantum_jump = np.random.normal(0, 0.1 * (self.upper_bound - self.lower_bound), self.dim)
            self.position += quantum_jump
            self.position = np.clip(self.position, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        self.initialize()

        for _ in range(self.budget):
            self.update_position()
            self.quantum_influence()
            fitness = self.evaluate(func, self.position)

            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_position = np.copy(self.position)

        return self.best_fitness, self.best_position
