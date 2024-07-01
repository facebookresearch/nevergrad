import numpy as np


class AdaptiveRefinementPSO:
    def __init__(self, budget=10000, population_size=50, omega=0.6, phi_p=0.2, phi_g=0.3, adapt_factor=0.05):
        self.budget = budget
        self.population_size = population_size
        self.omega = omega  # Inertia weight
        self.phi_p = phi_p  # Personal coefficient
        self.phi_g = phi_g  # Global coefficient
        self.dim = 5  # Dimension of the problem
        self.adapt_factor = adapt_factor  # Adaptation factor for dynamic parameter adjustment

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

        # Optimization loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Dynamic adjustment of inertia weight
                current_phase = evaluations / self.budget
                dynamic_omega = self.omega * (1 - current_phase) + self.adapt_factor * current_phase

                # Update velocity and position
                velocity[i] = (
                    dynamic_omega * velocity[i]
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

            # Dynamic adjustment of learning factors
            if evaluations % 1000 == 0:
                self.phi_p += self.adapt_factor * (1 - self.phi_p)  # Increase exploration
                self.phi_g -= self.adapt_factor * self.phi_g  # Decrease exploitation smoothness

            # Logging for monitoring
            if evaluations % 1000 == 0:
                print(f"Evaluation: {evaluations}, Best Fitness: {global_best_fitness}")

        return global_best_fitness, global_best
