import numpy as np


class QuantumSwarmOptimization:
    def __init__(
        self, budget=10000, num_particles=10, inertia_weight=0.5, cognitive_weight=1.5, social_weight=2.0
    ):
        self.budget = budget
        self.num_particles = num_particles
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.dim = 5
        self.search_space = (-5.0, 5.0)
        self.particles = np.random.uniform(
            self.search_space[0], self.search_space[1], (self.num_particles, self.dim)
        )
        self.velocities = np.zeros((self.num_particles, self.dim))
        self.personal_bests = self.particles.copy()
        self.personal_best_values = np.full(self.num_particles, np.Inf)
        self.global_best = None
        self.global_best_value = np.Inf

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.num_particles):
                current_position = self.particles[i]
                velocity_term1 = self.inertia_weight * self.velocities[i]
                velocity_term2 = (
                    self.cognitive_weight * np.random.rand() * (self.personal_bests[i] - current_position)
                )
                velocity_term3 = (
                    self.social_weight
                    * np.random.rand()
                    * (self.global_best - current_position if self.global_best is not None else 0)
                )
                new_velocity = velocity_term1 + velocity_term2 + velocity_term3
                new_position = current_position + new_velocity

                new_position = np.clip(new_position, self.search_space[0], self.search_space[1])

                f = func(new_position)
                if f < self.personal_best_values[i]:
                    self.personal_bests[i] = new_position
                    self.personal_best_values[i] = f

                if f < self.global_best_value:
                    self.global_best = new_position
                    self.global_best_value = f

                self.velocities[i] = new_velocity
                self.particles[i] = new_position

        return self.global_best_value, self.global_best
