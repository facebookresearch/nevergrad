import numpy as np


class AdaptiveInertiaParticleSwarmOptimization:
    def __init__(self, budget=10000, population_size=40, omega_max=0.9, omega_min=0.4, phi_p=0.2, phi_g=0.5):
        self.budget = budget
        self.population_size = population_size
        self.omega_max = omega_max  # Maximum inertia weight
        self.omega_min = omega_min  # Minimum inertia weight
        self.phi_p = phi_p  # Personal coefficient
        self.phi_g = phi_g  # Global coefficient
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([func(p) for p in particles])

        global_best = particles[np.argmin(personal_best_fitness)]
        global_best_fitness = np.min(personal_best_fitness)

        evaluations = self.population_size

        # Optimization loop
        while evaluations < self.budget:
            omega = self.omega_max - (self.omega_max - self.omega_min) * (evaluations / self.budget)
            for i in range(self.population_size):
                # Update velocity and position of particles
                velocity[i] = (
                    omega * velocity[i]
                    + self.phi_p * np.random.rand(self.dim) * (personal_best[i] - particles[i])
                    + self.phi_g * np.random.rand(self.dim) * (global_best - particles[i])
                )
                particles[i] += velocity[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate particle's fitness
                current_fitness = func(particles[i])
                evaluations += 1

                # Update personal and global bests
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best = particles[i]
                        global_best_fitness = current_fitness

            # Dynamic update of learning coefficients
            self.phi_p = max(0.1, self.phi_p - 0.5 * evaluations / self.budget)
            self.phi_g = min(0.6, self.phi_g + 0.5 * evaluations / self.budget)

        return global_best_fitness, global_best
