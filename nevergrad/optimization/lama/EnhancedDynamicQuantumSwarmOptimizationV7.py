import numpy as np


class EnhancedDynamicQuantumSwarmOptimizationV7:
    def __init__(
        self,
        budget=10000,
        num_particles=100,
        inertia_weight=0.5,
        cognitive_weight=1.5,
        social_weight=1.0,
        boundary_handling=True,
        step_size=0.1,
        mutation_rate=0.1,
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
        self.step_size = step_size
        self.mutation_rate = mutation_rate

    def update_parameters(self, iteration):
        pass

    def update_velocity_position(self, i, func):
        current_position = self.particles[i]
        r1, r2 = np.random.rand(), np.random.rand()
        velocity_term1 = self.inertia_weight * self.velocities[i]
        velocity_term2 = self.cognitive_weight * r1 * (self.personal_bests[i] - current_position)
        velocity_term3 = (
            self.social_weight
            * r2
            * (self.global_best - current_position if self.global_best is not None else 0)
        )
        new_velocity = velocity_term1 + velocity_term2 + velocity_term3
        new_position = current_position + new_velocity

        if self.boundary_handling:
            new_position = np.clip(new_position, self.search_space[0], self.search_space[1])

        # Mutation step to introduce diversity
        if np.random.rand() < self.mutation_rate:
            mutation_vector = np.random.normal(0, 0.1, self.dim)
            new_position = np.clip(new_position + mutation_vector, self.search_space[0], self.search_space[1])

        f = func(new_position)
        if f < self.personal_best_values[i]:
            self.personal_bests[i] = new_position
            self.personal_best_values[i] = f

        if f < self.global_best_value:
            self.global_best = new_position
            self.global_best_value = f

        self.velocities[i] = new_velocity
        self.particles[i] = new_position

    def __call__(self, func):
        for i in range(self.budget):
            for i in range(self.num_particles):
                self.update_velocity_position(i, func)

        return self.global_best_value, self.global_best
