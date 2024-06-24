import numpy as np


class EnhancedAdvancedQuantumSwarmOptimizationV2:
    def __init__(
        self,
        budget=10000,
        num_particles=30,
        inertia_weight=0.7,
        cognitive_weight=1.0,
        social_weight=1.0,
        damping=0.9,
        step_size=0.1,
        boundary=5.0,
    ):
        self.budget = budget
        self.dim = 5
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.damping = damping
        self.step_size = step_size
        self.best_fitness = np.inf
        self.best_position = None
        self.boundary = boundary

    def initialize_particles(self):
        self.particles_position = np.random.uniform(
            -self.boundary, self.boundary, (self.num_particles, self.dim)
        )
        self.particles_velocity = np.zeros((self.num_particles, self.dim))
        self.particles_best_position = self.particles_position.copy()
        self.particles_best_fitness = np.full(self.num_particles, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf

    def update_particles(self, func):
        for i in range(self.num_particles):
            fitness = func(self.particles_position[i])
            if fitness < self.particles_best_fitness[i]:
                self.particles_best_fitness[i] = fitness
                self.particles_best_position[i] = self.particles_position[i].copy()

            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = self.particles_position[i].copy()

            inertia_term = self.inertia_weight * self.particles_velocity[i]
            cognitive_term = (
                self.cognitive_weight
                * np.random.rand()
                * (self.particles_best_position[i] - self.particles_position[i])
            )
            social_term = (
                self.social_weight
                * np.random.rand()
                * (self.global_best_position - self.particles_position[i])
            )

            self.particles_velocity[i] = self.damping * (
                self.particles_velocity[i] + self.step_size * (inertia_term + cognitive_term + social_term)
            )
            self.particles_position[i] += self.particles_velocity[i]
            self.particles_position[i] = np.clip(self.particles_position[i], -self.boundary, self.boundary)

    def adapt_parameters(self):
        self.step_size = np.clip(self.step_size + 0.005 * np.random.randn(), 0.05, 0.3)
        self.damping = np.clip(self.damping - 0.005 * np.random.randn(), 0.7, 0.95)
        self.inertia_weight = np.clip(self.inertia_weight + 0.01 * np.random.randn(), 0.1, 1.0)
        self.cognitive_weight = np.clip(self.cognitive_weight + 0.01 * np.random.randn(), 0.1, 1.0)
        self.social_weight = np.clip(self.social_weight + 0.01 * np.random.randn(), 0.1, 1.0)

    def __call__(self, func):
        self.best_fitness = np.inf
        self.best_position = None
        self.global_best_position = None
        self.global_best_fitness = np.inf

        self.initialize_particles()

        for _ in range(1, self.budget):  # Start from 1 to avoid index out of bounds
            self.update_particles(func)
            self.adapt_parameters()

        self.best_fitness = self.global_best_fitness
        self.best_position = self.global_best_position

        return self.best_fitness, self.best_position
