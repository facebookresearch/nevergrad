import numpy as np


class EnhancedAdaptiveInertiaHybridOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, particles=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.particles = particles
        self.velocity_coeff = 0.7
        self.global_coeff = np.linspace(0.9, 0.5, self.budget)  # Dynamic adaptation of global coefficient
        self.local_coeff = np.linspace(0.5, 0.9, self.budget)  # Dynamic adaptation of local coefficient
        self.inertia_base = 1.1
        self.inertia_reduction = 0.3

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
            inertia_weight = self.inertia_base - (self.inertia_reduction * (evaluations / self.budget))

            for i in range(self.particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (
                    inertia_weight * velocities[i]
                    + self.local_coeff[evaluations] * r1 * (personal_best_positions[i] - positions[i])
                    + self.global_coeff[evaluations] * r2 * (global_best_position - positions[i])
                )

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.bounds[0], self.bounds[1])

                current_fitness = func(positions[i])
                evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = current_fitness

                if current_fitness < global_best_fitness:
                    global_best_position = positions[i]
                    global_best_fitness = current_fitness

                if evaluations >= self.budget:
                    break

        return global_best_fitness, global_best_position

    def __call__(self, func):
        return self.optimize(func)
