import numpy as np


class EnhancedMultiFocalAdaptiveOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, particles=100):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.particles = particles
        self.global_decay = 0.99
        self.local_decay = 0.95
        self.initial_velocity_scale = 0.1
        self.learning_rate = 0.3  # Added learning rate for adapting velocity updates

    def initialize_particles(self):
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.particles, self.dimension))
        velocities = np.random.randn(self.particles, self.dimension) * self.initial_velocity_scale
        return positions, velocities

    def evaluate_particles(self, func, positions):
        fitness = np.array([func(pos) for pos in positions])
        return fitness

    def optimize(self, func):
        positions, velocities = self.initialize_particles()
        fitness = self.evaluate_particles(func, positions)

        best_global_position = positions[np.argmin(fitness)]
        best_global_fitness = np.min(fitness)

        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()

        evaluations = self.particles

        while evaluations < self.budget:
            for i in range(self.particles):
                r1, r2 = np.random.rand(), np.random.rand()
                velocities[i] = (
                    self.global_decay * velocities[i]
                    + self.learning_rate * r1 * (personal_best_positions[i] - positions[i])
                    + self.learning_rate * r2 * (best_global_position - positions[i])
                )

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                new_fitness = func(positions[i])
                evaluations += 1

                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
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
