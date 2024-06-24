import numpy as np


class AdvancedFocusedAdaptiveOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, particles=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.particles = particles
        self.global_influence = 0.8  # Adjusted global influence
        self.local_influence = 0.2  # Adjusted local influence
        self.vel_scale = 0.1  # Adjusted velocity scaling for stabilization
        self.learning_factor = 0.5  # New: Doubling initial learning factor to speed up convergence

    def initialize_particles(self):
        # Initialize positions in the search space and velocities
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.particles, self.dimension))
        velocities = np.zeros_like(positions)
        return positions, velocities

    def evaluate_particles(self, func, positions):
        fitness = np.array([func(pos) for pos in positions])
        return fitness

    def optimize(self, func):
        positions, velocities = self.initialize_particles()
        fitness = self.evaluate_particles(func, positions)

        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()

        best_global_position = positions[np.argmin(fitness)]
        best_global_fitness = np.min(fitness)

        evaluations = self.particles

        while evaluations < self.budget:
            for i in range(self.particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    self.vel_scale * velocities[i]
                    + self.learning_factor * r1 * (personal_best_positions[i] - positions[i])
                    + self.learning_factor * r2 * (best_global_position - positions[i])
                )

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                new_fitness = func(positions[i])
                evaluations += 1

                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = new_fitness

                if new_fitness < best_global_fitness:
                    best_global_position = positions[i]
                    best_global_fitness = new_fitness

                if evaluations >= self.budget:
                    break

        return best_global_fitness, best_global_position

    def __call__(self, func):
        return self.optimize(func)
