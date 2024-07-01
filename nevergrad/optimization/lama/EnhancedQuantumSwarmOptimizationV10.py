import numpy as np


class EnhancedQuantumSwarmOptimizationV10:
    def __init__(
        self,
        budget=10000,
        num_particles=30,
        inertia_weight=0.7,
        cognitive_weight=1.0,
        social_weight=1.0,
        step_size=1.0,
        step_size_reduction=0.995,
        damping=0.9,
    ):
        self.budget = budget
        self.dim = 5
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.step_size = step_size
        self.step_size_reduction = step_size_reduction
        self.damping = damping
        self.best_fitness = np.inf
        self.best_position = None
        self.particles = []

    def initialize_particles(self):
        for _ in range(self.num_particles):
            particle = {
                "position": np.random.uniform(-5.0, 5.0, self.dim),
                "velocity": np.random.uniform(-1.0, 1.0, self.dim),
                "best_position": None,
                "best_fitness": np.inf,
            }
            self.particles.append(particle)

    def update_particle(self, particle, func):
        fitness = func(particle["position"])
        if fitness < particle["best_fitness"]:
            particle["best_fitness"] = fitness
            particle["best_position"] = particle["position"].copy()
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = particle["position"].copy()

        inertia_term = self.inertia_weight * particle["velocity"]
        cognitive_term = (
            self.cognitive_weight * np.random.rand() * (particle["best_position"] - particle["position"])
        )
        social_term = self.social_weight * np.random.rand() * (self.best_position - particle["position"])

        particle["velocity"] = self.damping * (inertia_term + self.step_size * (cognitive_term + social_term))
        particle["position"] += particle["velocity"]

    def adapt_parameters(self):
        self.step_size *= self.step_size_reduction

    def __call__(self, func):
        self.best_fitness = np.inf
        self.best_position = None
        self.particles = []

        self.initialize_particles()

        for _ in range(self.budget):
            for particle in self.particles:
                self.update_particle(particle, func)
            self.adapt_parameters()

        return self.best_fitness, self.best_position
