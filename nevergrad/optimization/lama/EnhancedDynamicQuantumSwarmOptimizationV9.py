import numpy as np


class EnhancedDynamicQuantumSwarmOptimizationV9:
    def __init__(
        self,
        budget=10000,
        num_particles=100,
        max_inertia_weight=0.9,
        min_inertia_weight=0.4,
        max_cognitive_weight=1.0,
        min_cognitive_weight=0.4,
        max_social_weight=1.0,
        min_social_weight=0.4,
        boundary_handling=True,
        alpha=0.5,
        delta=0.3,
        decay_rate=0.95,
        max_step=0.3,
        exploration_rate=0.2,
        gamma=0.05,
        mutation_rate=0.1,
    ):
        self.budget = budget
        self.num_particles = num_particles
        self.max_inertia_weight = max_inertia_weight
        self.min_inertia_weight = min_inertia_weight
        self.max_cognitive_weight = max_cognitive_weight
        self.min_cognitive_weight = min_cognitive_weight
        self.max_social_weight = max_social_weight
        self.min_social_weight = min_social_weight
        self.alpha = alpha
        self.decay_rate = decay_rate
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
        self.delta = delta
        self.max_step = max_step
        self.exploration_rate = exploration_rate
        self.gamma = gamma
        self.mutation_rate = mutation_rate

    def update_parameters(self, iteration):
        self.inertia_weight = max(
            self.min_inertia_weight, self.max_inertia_weight * (self.decay_rate**iteration)
        )
        self.cognitive_weight = max(
            self.min_cognitive_weight, self.max_cognitive_weight * (self.decay_rate**iteration)
        )
        self.social_weight = max(
            self.min_social_weight, self.max_social_weight * (self.decay_rate**iteration)
        )

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
        new_position = (
            current_position
            + self.alpha * new_velocity
            + (1 - self.alpha) * np.random.uniform(-self.max_step, self.max_step, self.dim)
        )

        if self.boundary_handling:
            new_position = np.clip(new_position, self.search_space[0], self.search_space[1])

        if np.random.rand() < self.exploration_rate:
            new_position = np.random.uniform(self.search_space[0], self.search_space[1], self.dim)

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

        self.velocities[i] = self.delta * new_velocity + self.gamma * np.random.normal(0, 1, self.dim)
        self.particles[i] = new_position

    def __call__(self, func):
        for i in range(self.budget):
            self.update_parameters(i)
            for i in range(self.num_particles):
                self.update_velocity_position(i, func)

        return self.global_best_value, self.global_best
