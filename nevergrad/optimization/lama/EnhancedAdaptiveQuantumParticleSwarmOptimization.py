import numpy as np


class EnhancedAdaptiveQuantumParticleSwarmOptimization:
    def __init__(
        self,
        budget=10000,
        num_particles=20,
        inertia_weight=0.5,
        cognitive_weight=1.5,
        social_weight=1.5,
        quantum_param=0.5,
        adapt_param=0.1,
        explore_rate=0.1,
    ):
        self.budget = budget
        self.num_particles = num_particles
        self.dim = 5
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.quantum_param = quantum_param
        self.adapt_param = adapt_param
        self.explore_rate = explore_rate

    def initialize_particles(self, func):
        self.particles = np.random.uniform(-5.0, 5.0, (self.num_particles, self.dim))
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_values = np.array([func(p) for p in self.particles])
        self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_values)]
        self.global_best_value = np.min(self.personal_best_values)
        self.adaptive_quantum_param = np.full(self.num_particles, self.quantum_param)

    def update_particles(self, func):
        for i in range(self.num_particles):
            r1, r2 = np.random.uniform(0, 1, 2)
            self.velocities[i] = (
                self.inertia_weight * self.velocities[i]
                + self.cognitive_weight * r1 * (self.personal_best_positions[i] - self.particles[i])
                + self.social_weight * r2 * (self.global_best_position - self.particles[i])
            )

            # Exploration mechanism
            exploration = np.random.uniform(-self.explore_rate, self.explore_rate, self.dim)
            self.particles[i] = np.clip(self.particles[i] + self.velocities[i] + exploration, -5.0, 5.0)

            new_value = func(self.particles[i])

            if new_value < self.personal_best_values[i]:
                self.personal_best_values[i] = new_value
                self.personal_best_positions[i] = self.particles[i]

            if new_value < self.global_best_value:
                self.global_best_value = new_value
                self.global_best_position = self.particles[i]

            # Adaptive Quantum Parameter update
            self.adaptive_quantum_param[i] = max(self.adaptive_quantum_param[i] - self.adapt_param, 0.1)

            # Quantum-inspired velocity update
            self.velocities[i] = self.adaptive_quantum_param[i] * self.velocities[i]

    def __call__(self, func):
        self.initialize_particles(func)

        for _ in range(self.budget // self.num_particles):
            self.update_particles(func)

        return self.global_best_value, self.global_best_position
