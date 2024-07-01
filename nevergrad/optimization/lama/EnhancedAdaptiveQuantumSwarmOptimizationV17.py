import numpy as np


class EnhancedAdaptiveQuantumSwarmOptimizationV17:
    def __init__(
        self,
        budget=10000,
        num_particles=30,
        inertia_weight=0.7,
        cognitive_weight=1.0,
        social_weight=1.0,
        damping=0.9,
        step_size=0.2,
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
        self.particles_position = np.random.uniform(-boundary, boundary, (num_particles, self.dim))
        self.particles_velocity = np.zeros((num_particles, self.dim))
        self.particles_best_position = self.particles_position.copy()
        self.particles_best_fitness = np.full(num_particles, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.boundary = boundary

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
                inertia_term + self.step_size * (cognitive_term + social_term)
            )
            self.particles_position[i] += self.particles_velocity[i]
            self.particles_position[i] = np.clip(self.particles_position[i], -self.boundary, self.boundary)

    def adapt_parameters(self):
        self.step_size *= 0.95
        self.damping *= 0.98

    def __call__(self, func):
        self.best_fitness = np.inf
        self.best_position = None
        self.global_best_position = None
        self.global_best_fitness = np.inf

        for _ in range(self.budget):
            self.update_particles(func)
            self.adapt_parameters()

        self.best_fitness = self.global_best_fitness
        self.best_position = self.global_best_position

        return self.best_fitness, self.best_position
