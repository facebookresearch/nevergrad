import numpy as np


class OptimizedRefinedAdaptiveRefinementPSO:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        omega_start=0.9,
        omega_end=0.4,
        phi_p=0.5,
        phi_g=0.8,
        phi_l=0.03,
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_start = omega_start  # Initial inertia weight
        self.omega_end = omega_end  # Final inertia weight
        self.phi_p = phi_p  # Personal coefficient
        self.phi_g = phi_g  # Global coefficient
        self.phi_l = phi_l  # Local neighborhood influence coefficient
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        # Initialize particles
        particles = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = particles.copy()
        personal_best_fitness = np.array([func(p) for p in particles])

        global_best = particles[np.argmin(personal_best_fitness)]
        global_best_fitness = min(personal_best_fitness)

        evaluations = self.population_size

        # Create neighborhood topology
        neighborhood_size = int(np.ceil(self.population_size * 0.1))
        neighbors = [
            np.random.choice(list(set(range(self.population_size)) - {i}), neighborhood_size, replace=False)
            for i in range(self.population_size)
        ]

        # Optimization loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Linearly decreasing inertia weight
                dynamic_omega = self.omega_start - (self.omega_start - self.omega_end) * (
                    evaluations / self.budget
                )

                # Update velocity and position
                r_p = np.random.random(self.dim)
                r_g = np.random.random(self.dim)
                r_l = np.random.random(self.dim)

                # Calculate local best within the neighborhood
                local_best = neighbors[i][np.argmin(personal_best_fitness[neighbors[i]])]

                velocity[i] = (
                    dynamic_omega * velocity[i]
                    + self.phi_p * r_p * (personal_best[i] - particles[i])
                    + self.phi_g * r_g * (global_best - particles[i])
                    + self.phi_l * r_l * (personal_best[local_best] - particles[i])
                )

                particles[i] += velocity[i]
                particles[i] = np.clip(particles[i], lb, ub)

                # Evaluate particle's fitness
                current_fitness = func(particles[i])
                evaluations += 1

                if evaluations >= self.budget:
                    break

                # Update personal and global bests
                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = particles[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best = particles[i]
                        global_best_fitness = current_fitness

        return global_best_fitness, global_best
