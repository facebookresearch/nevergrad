import numpy as np


class ConcentricConvergenceOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, particles=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.particles = particles
        self.global_learning_rate = 0.1
        self.local_learning_rate = 0.2
        self.convergence_rate = 0.05

    def initialize_particles(self):
        positions = np.random.uniform(self.bounds[0], self.bounds[1], (self.particles, self.dimension))
        return positions

    def evaluate_particles(self, func, positions):
        fitness = np.array([func(pos) for pos in positions])
        return fitness

    def optimize(self, func):
        positions = self.initialize_particles()
        fitness = self.evaluate_particles(func, positions)

        global_best_position = positions[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)
        evaluations = self.particles

        while evaluations < self.budget:
            for i in range(self.particles):
                # Concentric updates focusing on both global best and individual refinement
                personal_vector = np.random.normal(0, self.local_learning_rate, self.dimension)
                global_vector = np.random.normal(0, self.global_learning_rate, self.dimension)

                # Move towards global best while exploring locally
                positions[i] += global_vector * (global_best_position - positions[i]) + personal_vector

                # Ensure particles stay within bounds
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                # Evaluate new position
                new_fitness = func(positions[i])
                evaluations += 1

                # Update personal and global bests
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    if new_fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = new_fitness

                # Reduce learning rates to increase convergence over time
                self.global_learning_rate *= 1 - self.convergence_rate
                self.local_learning_rate *= 1 - self.convergence_rate

                if evaluations >= self.budget:
                    break

        return global_best_fitness, global_best_position

    def __call__(self, func):
        return self.optimize(func)
