import numpy as np


class RefinedInertiaFocalOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, particles=30):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.particles = particles
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.focus_factor = 0.1  # Intensity of focusing on better regions over time

    def initialize_particles(self):
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.particles, self.dimension))
        velocities = np.zeros_like(positions)
        return positions, velocities

    def evaluate_particles(self, func, positions):
        fitness = np.array([func(pos) for pos in positions])
        return fitness

    def optimize(self, func):
        positions, velocities = self.initialize_particles()
        fitness = self.evaluate_particles(func, positions)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_position = positions[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        evaluations = self.particles

        while evaluations < self.budget:
            for i in range(self.particles):
                r1, r2 = np.random.rand(2)

                # Dynamic adjustment of inertia weight
                inertia_decay = (1 - (evaluations / self.budget)) ** self.focus_factor
                inertia_weight = self.inertia_weight * inertia_decay

                # Velocity update formula
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                    + self.social_coeff * r2 * (global_best_position - positions[i])
                )

                # Position update
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                new_fitness = func(positions[i])
                evaluations += 1

                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = new_fitness

                if new_fitness < global_best_fitness:
                    global_best_position = positions[i]
                    global_best_fitness = new_fitness

                if evaluations >= self.budget:
                    break

        return global_best_fitness, global_best_position

    def __call__(self, func):
        return self.optimize(func)
