import numpy as np


class OptimizedDynamicQuantumSwarmOptimization:
    def __init__(
        self,
        budget=10000,
        num_particles=20,
        inertia_weight=0.9,
        cognitive_weight=2.5,
        social_weight=1.5,
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
        # Update parameters dynamically with enhancements
        if iteration % 1000 == 0 and iteration > 0:
            best_value_avg = np.mean(self.personal_best_values)
            global_improvement = abs(self.global_best_value - best_value_avg) / self.global_best_value
            if global_improvement < 0.01:
                self.inertia_weight *= 0.9
                self.cognitive_weight *= 1.1
                self.social_weight *= 1.1

        self.inertia_weight = max(0.4, min(0.9, self.inertia_weight))
        self.cognitive_weight = max(1.5, min(2.5, self.cognitive_weight))
        self.social_weight = max(1.2, min(1.8, self.social_weight))

    def __call__(self, func):
        for i in range(self.budget):
            self.update_parameters(i)
            for i in range(self.num_particles):
                current_position = self.particles[i]
                velocity_term1 = self.inertia_weight * self.velocities[i]
                cognitive_component = (
                    self.cognitive_weight * np.random.rand() * (self.personal_bests[i] - current_position)
                )
                social_component = (
                    self.social_weight
                    * np.random.rand()
                    * (self.global_best - current_position if self.global_best is not None else 0)
                )
                new_velocity = velocity_term1 + cognitive_component + social_component
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
