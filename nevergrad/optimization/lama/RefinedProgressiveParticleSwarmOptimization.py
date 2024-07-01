import numpy as np


class RefinedProgressiveParticleSwarmOptimization:
    def __init__(
        self, budget=10000, population_size=50, omega_start=0.9, omega_end=0.4, phi_p=0.2, phi_g=0.5
    ):
        self.budget = budget
        self.population_size = population_size
        self.omega_start = omega_start  # Initial inertia weight
        self.omega_end = omega_end  # Final inertia weight
        self.phi_p = phi_p  # Personal learning coefficient
        self.phi_g = phi_g  # Global learning coefficient
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        velocity = np.zeros((self.population_size, self.dim))
        personal_best = pop.copy()
        fitness = np.array([func(ind) for ind in pop])
        personal_best_fitness = fitness.copy()

        # Initialize global best
        global_best_idx = np.argmin(fitness)
        global_best = pop[global_best_idx]

        evaluations = self.population_size

        # Main optimization loop
        while evaluations < self.budget:
            r_p = np.random.uniform(0, 1, (self.population_size, self.dim))
            r_g = np.random.uniform(0, 1, (self.population_size, self.dim))

            # Calculate dynamic inertia weight based on iteration progress
            progress_ratio = evaluations / self.budget
            omega = self.omega_start - (self.omega_start - self.omega_end) * progress_ratio

            # Update velocity and positions
            velocity = (
                omega * velocity
                + self.phi_p * r_p * (personal_best - pop)
                + self.phi_g * r_g * (global_best - pop)
            )
            pop = np.clip(pop + velocity, lb, ub)

            # Evaluate new positions
            for i in range(self.population_size):
                current_fitness = func(pop[i])
                evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best[i] = pop[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < fitness[global_best_idx]:
                        global_best_idx = i
                        global_best = pop[i]

            # Adaptive learning coefficients based on progress
            self.phi_p = max(0.1, self.phi_p - progress_ratio * 0.05)
            self.phi_g = min(0.6, self.phi_g + progress_ratio * 0.05)

        return fitness[global_best_idx], global_best
