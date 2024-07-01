import numpy as np


class DynamicQuantumSwarmOptimization:
    def __init__(
        self,
        budget=10000,
        num_particles=20,
        inertia_weight=0.6,
        cognitive_weight=1.7,
        social_weight=2.2,
        boundary_handling=True,
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
        self.personal_best_values = np.full(self.num_particles, np.inf)
        self.global_best = None
        self.global_best_value = np.inf
        self.boundary_handling = boundary_handling

    def update_parameters(self, iteration):
        # Update parameters dynamically
        self.inertia_weight = 0.9 - 0.5 * (iteration / self.budget)
        self.cognitive_weight = 2.5 - 1.5 * (iteration / self.budget)
        self.social_weight = 1.5 + 0.7 * (iteration / self.budget)

    def __call__(self, func):
        for i in range(self.budget):
            self.update_parameters(i)
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

                if self.boundary_handling:
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
