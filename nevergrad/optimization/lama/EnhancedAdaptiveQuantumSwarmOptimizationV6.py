import numpy as np


class EnhancedAdaptiveQuantumSwarmOptimizationV6:
    def __init__(
        self,
        budget=10000,
        num_particles=30,
        inertia_weight=0.7,
        cognitive_weight=1.0,
        social_weight=1.0,
        step_size=0.2,
        damping=0.9,
        boundary=5.0,
    ):
        self.budget = budget
        self.dim = 5
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.step_size = step_size
        self.damping = damping
        self.best_fitness = np.inf
        self.best_position = None
        self.particles = []
        self.boundary = boundary

    def initialize_particles(self):
        for _ in range(self.num_particles):
            particle = {
                "position": np.random.uniform(-self.boundary, self.boundary, self.dim),
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

        particle["position"] = np.clip(particle["position"], -self.boundary, self.boundary)

    def adapt_parameters(self):
        self.step_size *= 0.95
        self.damping *= 0.98

    def adapt_weights(self):
        self.inertia_weight *= 0.95
        self.cognitive_weight += 0.01
        self.social_weight += 0.01

    def adapt_num_particles(self, func):
        mean_fitness = np.mean([func(particle["position"]) for particle in self.particles])
        if mean_fitness > 0.8 * self.best_fitness:
            self.num_particles += 1
        elif mean_fitness < 0.2 * self.best_fitness and self.num_particles > 2:
            self.num_particles -= 1

    def adapt_step_size(self):
        self.step_size *= 0.99

    def adapt_damping(self):
        self.damping *= 0.99

    def adapt_parameters_adaptive(self, func):
        func_values = [
            func(particle["position"]) for particle in self.particles if particle["best_position"] is not None
        ]
        if func_values:
            best_func_value = min(func_values)
            for i, particle in enumerate(self.particles):
                if particle["best_position"] is not None and func_values[i] > best_func_value:
                    self.step_size *= 0.95
                    self.damping *= 0.98
                else:
                    self.step_size *= 1.05
                    self.damping *= 1.02

    def adaptive_update(self, func, particle):
        prev_position = particle["position"].copy()
        self.update_particle(particle, func)
        new_position = particle["position"]

        if func(new_position) >= func(prev_position):
            particle["position"] = prev_position
            particle["velocity"] *= -0.5

    def __call__(self, func):
        self.best_fitness = np.inf
        self.best_position = None
        self.num_particles = 30
        self.particles = []

        self.initialize_particles()

        for _ in range(self.budget):
            for particle in self.particles:
                self.adaptive_update(func, particle)

            self.adapt_parameters()
            self.adapt_weights()
            self.adapt_num_particles(func)
            self.adapt_step_size()
            self.adapt_damping()
            self.adapt_parameters_adaptive(func)

        return self.best_fitness, self.best_position
