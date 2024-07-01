import numpy as np


class MultiFocalAdaptiveOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, particles=100):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.particles = particles
        self.global_decay = 0.99  # Slow decay rate for global exploration
        self.local_decay = 0.95  # Faster decay rate for local exploration
        self.initial_velocity_scale = 0.1

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

        evaluations = self.particles

        while evaluations < self.budget:
            for i in range(self.particles):
                # Randomly decide whether to move towards global best or explore
                if np.random.rand() > 0.5:
                    velocities[i] += (best_global_position - positions[i]) * np.random.rand()
                else:
                    velocities[i] += np.random.normal(0, 1, self.dimension) * self.initial_velocity_scale

                # Update position
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                new_fitness = func(positions[i])
                evaluations += 1

                # Update personal and global bests
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < best_global_fitness:
                        best_global_position = positions[i]
                        best_global_fitness = new_fitness

                # Decaying velocity magnitudes over iterations to increase exploitation
                velocities[i] *= self.global_decay

                if evaluations >= self.budget:
                    break

        return best_global_fitness, best_global_position

    def __call__(self, func):
        return self.optimize(func)
