import numpy as np


class RefinedQuantumNesterovSynergyV2:
    def __init__(
        self,
        budget,
        dim=5,
        learning_rate=0.1,
        momentum=0.95,
        quantum_influence_rate=0.25,
        adaptive_lr_factor=0.95,
        elite_fraction=0.35,
        noise_factor=0.25,
        perturbation_intensity=0.1,
    ):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.quantum_influence_rate = quantum_influence_rate
        self.adaptive_lr_factor = adaptive_lr_factor
        self.elite_fraction = elite_fraction
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.noise_factor = noise_factor
        self.perturbation_intensity = perturbation_intensity

    def initialize(self):
        self.population = np.random.uniform(
            self.lower_bound, self.upper_bound, (int(self.budget * self.elite_fraction), self.dim)
        )
        self.velocities = np.zeros((int(self.budget * self.elite_fraction), self.dim))
        self.fitnesses = np.full(int(self.budget * self.elite_fraction), np.inf)

    def evaluate_population(self, func):
        for i in range(len(self.population)):
            fitness = func(self.population[i])
            if fitness < self.fitnesses[i]:
                self.fitnesses[i] = fitness

    def update_particles(self):
        best_idx = np.argmin(self.fitnesses)
        global_best = self.population[best_idx]

        for i in range(len(self.population)):
            if np.random.rand() < self.quantum_influence_rate:
                quantum_jump = np.random.normal(0, self.perturbation_intensity, self.dim)
                self.population[i] += quantum_jump * (global_best - self.population[i])

            noise = np.random.normal(0, self.noise_factor, self.dim)
            self.velocities[i] = self.momentum * self.velocities[i] - self.learning_rate * noise
            future_position = self.population[i] + self.momentum * self.velocities[i]
            future_position = np.clip(future_position, self.lower_bound, self.upper_bound)
            self.population[i] = future_position

        self.learning_rate *= self.adaptive_lr_factor

    def __call__(self, func):
        self.initialize()
        total_evaluations = len(self.population)
        while total_evaluations < self.budget:
            self.evaluate_population(func)
            self.update_particles()
            total_evaluations += len(self.population)

        best_idx = np.argmin(self.fitnesses)
        return self.fitnesses[best_idx], self.population[best_idx]
